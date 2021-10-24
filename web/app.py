from pprint import pprint
import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
import pylab, random, os, math, sklearn
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
from flask import Flask, request, jsonify, make_response, render_template
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point, Polygon, shape
import folium
from folium.plugins import HeatMap

app = Flask(__name__)
# CORS(app, resources=r'/*')

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def getx(self):
        return self.x

    def gety(self):
        return self.y


class Getlen:
    def __init__(self, p1, p2):
        self.x = p1.getx() - p2.getx()
        self.y = p1.gety() - p2.gety()
        self.len = math.sqrt((self.x ** 2) + (self.y ** 2))

    def getlen(self):
        return self.len


# def save_single_image(img_path, save_path, soup):
#     # extract crack name and position
#     names = []
#     boxes = []
#     for crack in soup.find_all("object"):
#         names.append(crack.find("name").text)
#         box = crack.bndbox
#         box = {
#             'xmin': int(box.xmin.text),
#             'xmax': int(box.xmax.text),
#             'ymin': int(box.ymin.text),
#             'ymax': int(box.ymax.text)
#         }
#         boxes.append(box)
#
#     # read image
#     filename = soup.filename.text
#     img = img_to_array(load_img(os.path.join(img_path, filename))).astype(int)
#
#     # annotate the image
#     fig, ax = plt.subplots(figsize=(10, 20))
#     ax.imshow(img)
#     for name, box in zip(names, boxes):
#         x = [box["xmin"]] + [box["xmax"]] * 2 + [box["xmin"]] * 2
#         y = [box["ymin"]] * 2 + [box["ymax"]] * 2 + [box["ymin"]]
#         ax.plot(x, y, c="red", linewidth="2")
#         ax.text(box["xmax"] + 50, 0.5 * (box["ymin"] + box["ymax"]), name, c="red", fontsize=20)
#
#     # save image
#     fig.savefig(os.path.join(save_path, filename))


@app.route("/")
def aa():
    a = make_response(render_template("sample.html"))
    return a


@app.route("/sample.html")
def display_sample():
    a = make_response(render_template("sample.html"))
    return a

@app.route("/up.html")
def display_upload():
    a = make_response(render_template("up.html"))
    return a

@app.route("/Dashboard.html")
def display_sta():
    a = make_response(render_template("Dashboard.html"))
    return a

@app.route("/map5.html", methods=['GET'])
def aaa():
    a = make_response(render_template("map5.html"))
    return a


@app.route("/heat_map.html", methods=['GET'])
def aaaa():
    a = make_response(render_template("heat_map.html"))
    return a


@app.route("/heat_map_style2.html", methods=['GET'])
def aaaaa():
    a = make_response(render_template("heat_map_style2.html"))
    return a


@app.route("/show_map", methods=['GET'])
def dd():
    a = make_response(render_template("map5.html"))
    return a


@app.route("/show_heat_map", methods=['GET'])
def ddd():
    a = make_response(render_template("heat_map.html"))
    return a


@app.route("/show_heat_map_style2", methods=['GET'])
def dddd():
    a = make_response(render_template("heat_map_style2.html"))
    return a


@app.route("/upload", methods=['GET', 'POST'])
def Img():
    if request.method == "GET":
        print(1)
        res = make_response(render_template('aa.html'))
        return res
    elif request.method == "POST":
        if "myImg" in request.files:
            ge = request.files.get("myImg")
            fname = ge.filename
            path = "./img/" + fname
            ge.save(path)
            return "saved"


# @app.route('/upload_f', methods=['POST'])
# def upload():
#     pprint(request.files.getlist("file"))
#     pprint(request.files.getlist("file")[2].filename)
#     return "upload success"


@app.route("/upload_ann", methods=['GET', 'POST'])
def file():
    if request.method == "GET":
        res = make_response(render_template('aa.html'))
        return res
    elif request.method == "POST":
        if "file" in request.files:
            ge = request.files.get("file")
            fname = ge.filename
            path = "./img/" + fname
            ge.save(path)
            return "saved"


@app.route("/draw_map", methods=['GET'])
def draw():
    df = pd.read_csv('img/images.csv')

    fmap = folium.Map(location=[-37.8136, 144.9631], zoom_start=10)
    for i, row in df.iterrows():
        if i % 20 != 0:
            continue
        file_name = '/file/anno-image/' + row['image_name']
        if row["crack_1"] == "none" and row["crack_2"] == "none" and row["crack_3"] == "none":
            continue

            # fmap.add_child(folium.Marker(location=[row['Latitude'], row['Longitude']],
            #                              popup=f'<img class = "aaa" src={file_name} alt = \'picture\' width=auto height=auto />', icon=folium.Icon(color='green')))

        else:
            fmap.add_child(folium.Marker(location=[row['Latitude'], row['Longitude']],
                                         popup=f'top:{row["crack_1"]}\nmiddle:{row["crack_2"]}\nbottom:{row["crack_3"]}',
                                         icon=folium.Icon(color='red')))

    name = "geo/vic_lga.shp"
    data_shp = geopandas.GeoDataFrame.from_file(name, encoding="utf-8")
    folium.GeoJson(
        data_shp.to_json(),
        style_function=lambda feature: {
            'fillColor': '#ffff00',
            'color': 'black',
            'weight': 2,
            'dashArray': '5, 5'
        },
        popup='ahaha',
    ).add_to(fmap)

    fmap.save('./templates/map5.html')
    return "down"


# def draw_map():
#   # 建立地圖與設定位置
#   fmap = folium.Map(location=[-37.8136, 144.9631], zoom_start=6)

@app.route("/draw_heat_map", methods=['GET'])
def draw_heat():
    name = "geo/vic_lga.shp"
    data_shp = geopandas.GeoDataFrame.from_file(name, encoding="utf-8")
    temp = eval(data_shp.to_json())
    max_x = 0
    min_x = 10000000
    max_y = -10000000
    min_y = 0
    n = 0
    di = {}
    for fe in temp['features']:
        x_total = 0
        y_total = 0
        n = 0
        for point in fe['geometry']['coordinates']:
            for i in point:
                x_total += i[0]
                y_total += i[1]
                n += 1
                if i[0] > max_x:
                    max_x = i[0]
                if i[0] < min_x:
                    min_x = i[0]
                if i[1] > max_y:
                    max_y = i[1]
                if i[1] < min_y:
                    min_y = i[1]
            x_middle = x_total / n
            y_middle = y_total / n
            di[(x_middle, y_middle)] = 0
    df = pd.read_csv('img/images.csv')
    for i, row in df.iterrows():
        if row["crack_1"] == "none" and row["crack_2"] == "none" and row["crack_3"] == "none":
            continue
        mi = 1000000000000
        p1 = Point(row['Latitude'], row['Longitude'])

        for p in di.keys():
            p2 = Point(p[1], p[0])
            l = Getlen(p1, p2)
            d = l.getlen()
            if d < mi:
                mi = d
                mi_p = p
        try:
            di[mi_p] += 1
        except:
            continue
    heat = [[i[1], i[0], di[i]] for i in di]
    fmap = folium.Map(location=[-37.8136, 144.9631], zoom_start=10)
    folium.GeoJson(
        data_shp.to_json(),
        style_function=lambda feature: {
            'fillColor': '#ffff00',
            'color': 'black',
            'weight': 2,
            'dashArray': '5, 5'
        },
        popup='ahaha',
    ).add_to(fmap)
    HeatMap(heat).add_to(fmap)
    fmap.save('./templates/heat_map.html')
    return "heat map down"


@app.route("/draw_heat_map_style2", methods=['GET'])
def draw_heat_style2():
    name = "geo/vic_lga.shp"
    data_shp = geopandas.GeoDataFrame.from_file(name, encoding="utf-8")
    temp = eval(data_shp.to_json())
    max_x = 0
    min_x = 10000000
    max_y = -10000000
    min_y = 0

    n = 0
    di = {}
    qq = {}
    for fe in temp['features']:
        x_total = 0
        y_total = 0
        n = 0
        name = fe['properties']['ABB_NAME']

        for point in fe['geometry']['coordinates']:
            for i in point:
                x_total += i[0]
                y_total += i[1]
                n += 1
                if i[0] > max_x:
                    max_x = i[0]
                if i[0] < min_x:
                    min_x = i[0]
                if i[1] > max_y:
                    max_y = i[1]
                if i[1] < min_y:
                    min_y = i[1]
            x_middle = x_total / n
            y_middle = y_total / n
            di[(x_middle, y_middle, name)] = 0
            qq[name] = 0
    df = pd.read_csv('img/images.csv')
    for i, row in df.iterrows():
        if row["crack_1"] == "none" and row["crack_2"] == "none" and row["crack_3"] == "none":
            continue
        mi = 1000000000000
        p1 = Point(row['Latitude'], row['Longitude'])

        for p in di.keys():
            p2 = Point(p[1], p[0])
            l = Getlen(p1, p2)
            d = l.getlen()
            if d < mi:
                mi = d
                mi_p = p
        try:
            qq[mi_p[2]] += 1
        except:
            continue
    a = []
    for i in qq:
        a.append({"name": i, "num": qq[i]})
    mm = pd.DataFrame(a)
    fmap = folium.Map(location=[-37.8136, 144.9631], zoom_start=10)
    co = folium.Choropleth(geo_data=temp,
                           data=mm,
                           columns=['name', 'num'],
                           key_on='properties.ABB_NAME',
                           fill_color='YlOrRd',
                           fill_opacity=0.8,
                           line_opacity=0.2
                           ).add_to(fmap)
    co.geojson.add_child(
        folium.features.GeoJsonTooltip(mm['name'], labels=False))
    fmap.save("./templates/heat_map_style2.html")

    return "heat map2 down"


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
