import gradio as gr
import streamlit as st

from PIL import Image, ImageDraw, ImageFont
import scipy.io.wavfile as wavfile
import torch

from transformers import pipeline, AutoTokenizer, VitsModel


model_path = "facebook/detr-resnet-50"
tts_model_path = "kakao-enterprise/vits-ljs"


@st.cache_resource
def load_models():

    object_detector = pipeline(
        "object-detection",
        model=model_path
    )

    tts_tokenizer = AutoTokenizer.from_pretrained(
        tts_model_path
    )

    tts_model = VitsModel.from_pretrained(
        tts_model_path
    )

    return object_detector, tts_tokenizer, tts_model


object_detector, tts_tokenizer, tts_model = load_models()


def generate_audio(text):

    inputs = tts_tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = tts_model(
            **inputs
        ).waveform

    audio = output.squeeze().cpu().numpy()

    output_file = "output.wav"

    wavfile.write(
        output_file,
        rate=tts_model.config.sampling_rate,
        data=audio
    )

    return output_file


def read_objects(detection_objects):

    object_counts = {}

    for detection in detection_objects:

        label = detection["label"]

        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    if not object_counts:
        return (
            "This picture does not contain any "
            "recognizable objects."
        )

    plural_forms = {
        "person": "people",
        "child": "children",
        "man": "men",
        "woman": "women",
        "mouse": "mice",
        "sheep": "sheep",
        "tooth": "teeth"
    }

    descriptions = []

    for label, count in object_counts.items():

        if count == 1:

            descriptions.append(
                f"1 {label}"
            )

        else:

            plural = plural_forms.get(
                label,
                label + "s"
            )

            descriptions.append(
                f"{count} {plural}"
            )

    if len(descriptions) == 1:

        response = descriptions[0]

    elif len(descriptions) == 2:

        response = (
            descriptions[0]
            + " and "
            + descriptions[1]
        )

    else:

        response = (
            ", ".join(descriptions[:-1])
            + " and "
            + descriptions[-1]
        )

    return f"This picture contains {response}."


def draw_bounding_boxes(
    image,
    detections,
    font_path=None,
    font_size=20
):

    draw_image = image.copy()

    draw = ImageDraw.Draw(
        draw_image
    )

    if font_path:

        font = ImageFont.truetype(
            font_path,
            font_size
        )

    else:

        font = ImageFont.load_default()

    for detection in detections:

        box = detection["box"]

        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]

        draw.rectangle(
            [
                (xmin, ymin),
                (xmax, ymax)
            ],
            outline="red",
            width=3
        )

        label = detection["label"]

        score = detection["score"]

        text = f"{label} {score:.2f}"

        text_size = draw.textbbox(
            (xmin, ymin),
            text,
            font=font
        )

        draw.rectangle(
            [
                (text_size[0], text_size[1]),
                (text_size[2], text_size[3])
            ],
            fill="red"
        )

        draw.text(
            (xmin, ymin),
            text,
            fill="white",
            font=font
        )

    return draw_image


def detect_object(image):

    if image is None:
        return None, None

    raw_image = image

    output = object_detector(
        raw_image,
        threshold=0.2
    )

    processed_image = draw_bounding_boxes(
        raw_image,
        output
    )

    natural_text = read_objects(
        output
    )

    print(natural_text)

    processed_audio = generate_audio(
        natural_text
    )

    return (
        processed_image,
        processed_audio
    )


demo = gr.Interface(

    fn=detect_object,

    inputs=[
        gr.Image(
            label="Select Image",
            type="pil"
        )
    ],

    outputs=[
        gr.Image(
            label="Processed Image",
            type="pil"
        ),
        gr.Audio(
            label="Generated Audio"
        )
    ],

    title="Object Detector with Audio",

    description=(
        "THIS APPLICATION WILL BE USED TO HIGHLIGHT OBJECTS "
        "AND GIVE AUDIO DESCRIPTION FOR THE PROVIDED INPUT IMAGE."
    )
)


st.set_page_config(
    page_title="Object Detector with Audio",
    page_icon="🔊",
    layout="centered"
)

st.title("Object Detector with Audio")

st.write(
    "Upload an image to detect objects, "
    "highlight them with bounding boxes, "
    "and generate an audio description."
)

uploaded_file = st.file_uploader(
    "Select Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("Detect Objects"):

        with st.spinner(
            "Detecting objects and generating audio..."
        ):

            processed_image, processed_audio = detect_object(
                image
            )

        st.subheader("Processed Image")

        st.image(
            processed_image,
            use_container_width=True
        )

        detections = object_detector(
            image,
            threshold=0.2
        )

        natural_text = read_objects(
            detections
        )

        st.subheader("Description")

        st.write(
            natural_text
        )

        st.subheader("Generated Audio")

        with open(
            processed_audio,
            "rb"
        ) as audio_file:

            audio_bytes = audio_file.read()

        st.audio(
            audio_bytes,
            format="audio/wav"
        )