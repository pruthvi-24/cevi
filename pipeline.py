import os
import sys
import numpy as np
import pandas as pd
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

from PIL import Image

# ==================================================
# PATH SETUP (ABSOLUTE – IMPORTANT FOR FASTAPI)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "mobilenetv3_large.tflite")
DISHES_CSV = os.path.join(BASE_DIR, "dishes.csv")
WATER_CSV = os.path.join(BASE_DIR, "water.csv")

IMG_SIZE = (224, 224)

# ==================================================
# CLASS LABELS (ORDER MUST MATCH TRAINING)
# ==================================================
CLASS_NAMES = [
    'achar','aloo gobi','aloo matar','aloo methi','aloo puri','aloo tikki',
    'appam','apple','apple pie','bagels','baingan bharta','banana','basundi',
    'beetroot','besan cheela','besan laddu','bhindi masala','biryani','boondi',
    'butter chicken','cabbage','canned potatoes','capsicum','carrots',
    'cauliflower','chai','chana masala','chapati','chicken rezala',
    'chicken tikka','chicken tikka masala','chilli pepper','chilli potato',
    'chole bhature','chop suey','chow mein','cooked oatmeal','cooked pasta',
    'corn','cucumber','dal makhani','dal tadka','dhokla','doughnut','dum aloo',
    'fried chicken','fried rice','gajar ka halwa','garlic','ginger',
    'gobi manchurian','grape','gujiya','idli','imarti','jalebi','kachori',
    'kadai paneer','kadhi pakoda','kaju katli','kalakand','kathi roll','kebabs',
    'khandvi','khichdi','kiwi','kofta','kulfi','lassi','lemon','lettuce',
    'litti chokha','malpua','masala dosa','medu vada','mishti doi','missi roti',
    'modak','momos','mysore pak','naan bread','navratan korma','omelette',
    'onion','onion pakoda','orange','palak paneer',
    'paneer butter masala','papad','paratha','pav bhaji','peanut chikki',
    'pear','peas','phirni','pineapple','poha','popcorn','rabri','rajma',
    'ras malai','rasgulla','rice cooked','samosa','sandwich','scrambled eggs',
    'shankarpali','sheer khurma','sheera','shelled soy bean','shrikhand',
    'spinach','spring rolls','sprouts','stuffed karela',
    'sunny side up eggs','sweet potatoes','taco','toast','tomato','turnip',
    'uttapam','vada pav','watermelon'
]

# ==================================================
# TFLITE INTERPRETER (LOADED ONCE)
# ==================================================
_interpreter = None
_input_details = None
_output_details = None

def get_interpreter():
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        _interpreter = tflite.Interpreter(model_path=MODEL_FILE)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
    return _interpreter, _input_details, _output_details

# ==================================================
# IMAGE PREPROCESSING (RAW INPUT – VERY IMPORTANT)
# ==================================================
def preprocess_pil_image(pil_image):
    img = pil_image.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x

# ==================================================
# DISH PREDICTION
# ==================================================
def predict_dish(pil_image):
    interpreter, input_details, output_details = get_interpreter()

    x = preprocess_pil_image(pil_image)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_details[0]["index"])[0]
    pred_idx = int(np.argmax(y))

    return CLASS_NAMES[pred_idx], float(y[pred_idx])

# ==================================================
# INGREDIENT LOOKUP
# ==================================================
def get_ingredients(dish):
    df = pd.read_csv(DISHES_CSV)
    df.columns = df.columns.str.strip()

    row = df[df['Dish (cleaned)'].str.lower() == dish.lower()]
    if row.empty:
        return []

    return [i.strip() for i in row['Matched_ingredients'].values[0].split(',')]

# ==================================================
# WATER FOOTPRINT CALCULATION
# ==================================================
def calculate_water_footprint(ingredients):
    df = pd.read_csv(WATER_CSV)
    df.columns = df.columns.str.strip()
    df['Item_lower'] = df['Item'].str.lower()

    g = b = gr = 0.0

    for ing in ingredients:
        r = df[df['Item_lower'] == ing.lower()]
        if not r.empty:
            g += float(r['Green (L/kg)'].values[0])
            b += float(r['Blue (L/kg)'].values[0])
            gr += float(r['Grey (L/kg)'].values[0])

    total = g + b + gr

    return {
        "green": round(g, 2),
        "blue": round(b, 2),
        "grey": round(gr, 2),
        "total": round(total, 2),
        "green_pct": round((g / total) * 100, 2) if total else 0,
        "blue_pct": round((b / total) * 100, 2) if total else 0,
        "grey_pct": round((gr / total) * 100, 2) if total else 0
    }

# ==================================================
# SUSTAINABILITY MESSAGE
# ==================================================
def water_message(g, b, gr):
    msg = []
    if g >= 50:
        msg.append("High reliance on rain-fed (green) water – environmentally positive.")
    if b >= 30:
        msg.append("Moderate to high freshwater (blue) water usage – consume responsibly.")
    if gr >= 10:
        msg.append("Noticeable pollution-related (grey) water footprint – avoid frequent consumption.")
    return msg

# ==================================================
# MAIN PIPELINE (USED BY FASTAPI)
# ==================================================
def analyze_food_image(pil_image):
    dish, conf = predict_dish(pil_image)
    ingredients = get_ingredients(dish)
    wf = calculate_water_footprint(ingredients)

    advice = water_message(
        wf["green_pct"],
        wf["blue_pct"],
        wf["grey_pct"]
    )

    return {
        "dish": dish,
        "confidence": round(conf * 100, 2),
        "ingredients": ingredients,
        "water_footprint": wf,
        "sustainability_advice": advice
    }

# ==================================================
# CLI SUPPORT (OPTIONAL)
# ==================================================
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "biryani.jpg"
    img = Image.open(img_path).convert("RGB")
    result = analyze_food_image(img)
    print(result)
