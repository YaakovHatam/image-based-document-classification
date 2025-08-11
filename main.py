from input_pipeline import pdf_to_images
from template_detection import template_detection_main


if __name__ == "__main__":
    pdf_to_images("./to_process/customer1.pdf", "./out/customer1")
    template_detection_main("./out/customer1", "./out/customer1")
