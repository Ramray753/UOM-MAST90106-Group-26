#!/usr/bin/env python3

import base64
import os.path
import numpy as np
import tornado.ioloop
import tornado.web
from io import BytesIO
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from PIL import Image

# Replace it with your image folder
image_path = os.path.realpath("../../data/LCMS-Range-Shuffled/train")
# Replace with your favorite hotkeys
hotkeys = ['Z'   , 'X'  , 'C'   , 'V'   , 'B'   , 'N'   ]
types   = ["NONE", "LAT", "LONG", "CROC", "DIAG", "RAIL"]

## Database
engine = create_engine("sqlite:///crack.db", echo=False, future=True)
Session = sessionmaker(bind = engine)
Base = declarative_base()
class Entry(Base):
    __tablename__ = 'images'
    id    = Column(Integer, primary_key=True)
    crack = Column(String)
    def create(index, region, crack):
        return Entry(id=(index*3+region), crack=crack)
    def query(session, index, region):
        return session.query(Entry).filter(Entry.id==(index*3+region))
    def query_range(session, start, number):
        return session.query(Entry).filter(Entry.id>=(start*3), Entry.id<((start+number)*3))
    def __repr__(self):
        return f"Entry(id={self.id!r}, index={self.id//3!r}, region={self.id%3!r}), crack={self.crack!r}"
Base.metadata.create_all(engine)


def read_image(index, region):
    file = os.path.realpath(os.path.join(image_path, f"{index:05d}.JPG"))
    if not os.path.exists(file):
        raise tornado.web.HTTPError(404)
    with Image.open(file) as im:
        img = np.array(im).astype(int)
        ideal_shape = (1230, 410, 3)
        crop_bound = {
            "xmin": int((img.shape[1] - ideal_shape[1]) / 2),
            "xmax": int((img.shape[1] + ideal_shape[1]) / 2 - 1),
            "ymin": int((img.shape[0] - ideal_shape[0]) / 2),
            "ymax": int((img.shape[0] + ideal_shape[0]) / 2 - 1)
        }
        img_cropped = img[crop_bound["ymin"]:crop_bound["ymax"] + 1, crop_bound["xmin"]:crop_bound["xmax"] + 1]
        y_space = np.linspace(0, ideal_shape[0], 4).astype(int)
        y_space[-1] += 1

        data = BytesIO()
        Image.fromarray(np.uint8(img_cropped[y_space[region]:y_space[region + 1], ])).save(data, "JPEG")
        data64 = base64.b64encode(data.getvalue())
        return u'data:img/jpeg;base64,'+data64.decode('utf-8')

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.render("index.html")

class CSVHandler(tornado.web.RequestHandler):
    def get(self, start, number):
        start = int(start)
        number = int(number)
        labeled = {}
        with Session() as session:
            for entry in Entry.query_range(session, start, number).all():
                index = entry.id // 3
                region = entry.id % 3
                labeled.setdefault(index, ['unknown', 'unknown', 'unknown'])[region] = entry.crack.decode().lower()
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.set_header("Content-Type", "text/csv")
        self.write("image_name,crack_1,crack_2,crack_3\r\n")
        for index in sorted(labeled.keys()):
            cracks = labeled[index]
            self.write(f"{index:05d}.JPG,{cracks[0]},{cracks[1]},{cracks[2]}\r\n")

class QueryHandler(tornado.web.RequestHandler):
    def get(self, start, number):
        start = int(start)
        number = int(number)
        with Session() as session:
            labeled = set(entry.id for entry in Entry.query_range(session, start, number).all())
        unlabeled = []
        for i in range(start, start+number):
            for k in [0,1,2]:
                if (i*3 + k) not in labeled:
                    unlabeled.append((i, k))
                    break
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.render("query.html", start=start, number=number, unlabeled=unlabeled)

class PrevHandler(tornado.web.RequestHandler):
    def get(self, index, region):
        prev = (int(index) * 3 + int(region)) - 1
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.redirect(f"/classify/{prev//3:d}/{prev%3:d}", permanent=True)

class NextHandler(tornado.web.RequestHandler):
    def get(self, index, region):
        prev = (int(index) * 3 + int(region)) + 1
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.redirect(f"/classify/{prev//3:d}/{prev%3:d}", permanent=True)

class ClassifyHandler(tornado.web.RequestHandler):
    def get(self, index, region):
        index = int(index)
        region = int(region)
        if index < 1:
            self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.redirect("/classify/1/0", permanent=True)
        else:
            image = read_image(index, region)
            crack = ""
            with Session() as session:
                result = Entry.query(session, index, region)
                if result.count() > 0:
                    crack = result.one().crack.decode()
            self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.render("classify.html", image=image, selection=zip(types, hotkeys, ["checked" if x==crack else "" for x in types]), crack=crack)
    def put(self, index, region):
        print(self.request.body)
        index = int(index)
        region = int(region)
        with Session() as session:
            result = Entry.query(session, index, region)
            if result.count() == 0:
                session.add(Entry.create(index, region, self.request.body))
            else:
                result.one().crack = self.request.body
            session.commit()
        self.set_header("Content-Type", "text/plain")
        self.write("OK")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/query/(\d+)/(\d+)/labeled.csv", CSVHandler     ),
        (r"/query/(\d+)/(\d+)"            , QueryHandler   ),
        (r"/classify/(\d+)/([012])"       , ClassifyHandler),
        (r"/classify/(\d+)/([012])/prev"  , PrevHandler    ),
        (r"/classify/(\d+)/([012])/next"  , NextHandler    ),
    ])

if __name__ == "__main__":
    print("Visit GUI at http://127.0.0.1:8888/,")
    print("or via your server's IP if accessing from a different device.")
    print("[Make sure that your firewall allows incoming traffic.]")
    app = make_app()
    app.listen(8888, '0.0.0.0')
    tornado.ioloop.IOLoop.current().start()
