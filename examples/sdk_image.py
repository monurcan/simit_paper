"""Text-to-image via the Python SDK.

Start a server first, e.g.:  mstar serve bagel
"""

from mstar import MStarClient

client = MStarClient("http://localhost:8000")

png = client.generate_image("a cat in a hat, oil painting", think_mode=True)
with open("out.png", "wb") as f:
    f.write(png)
print(f"wrote out.png — {len(png)} bytes")
