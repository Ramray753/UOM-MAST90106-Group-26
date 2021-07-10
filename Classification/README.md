# A web-based classification tool

### Required packages

```
sudo pip install numpy
sudo pip install pillow
sudo pip install tornado
sudo pip install SQLAlchemy
```

### How to run?

Edit following lines in `main.py` accordingly:
```python3
# Replace it with your image folder
image_path = os.path.realpath("../../data/LCMS-Range-Shuffled/train")
# Replace with your favorite hotkeys
hotkeys = ['Z'   , 'X'  , 'C'   , 'V'   , 'B'   , 'N'   ]
```

Then execute:
```bash
python3 main.py
```

Finally, open a web browser and go to http://127.0.0.1:8888/.
