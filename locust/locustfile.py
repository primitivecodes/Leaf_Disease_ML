import glob, io, random
from locust import HttpUser, task, between
from PIL import Image

files = glob.glob("data/test/*/*.jpg")
def sample():
    if files:
        return open(random.choice(files), "rb").read()
    b = io.BytesIO(); Image.new("RGB", (160, 160), (30, 120, 40)).save(b, "JPEG"); return b.getvalue()

class User(HttpUser):
    wait_time = between(0.1, 0.5)
    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("leaf.jpg", sample(), "image/jpeg")})
    @task(1)
    def health(self):
        self.client.get("/health")
