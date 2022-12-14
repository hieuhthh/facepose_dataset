import gdown
import os

DES = '/home/lap14880/hieunmt/facepose/faceposelandmark/download'

try:
    os.mkdir(DES)
except:
    pass

url = "https://drive.google.com/file/d/1-ygCl6Uqr2YlSGDNRjxlpKbFjzCyacSu/view?usp=sharing"
output = f"{DES}/300W.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1HPJr5cPvZfky-VD9mmEgtXCc0_0luU7r/view?usp=sharing"
output = f"{DES}/afw.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1VYmrs-h5pIsrUSrBgqk1hInQ5qPXL3eG/view?usp=sharing"
output = f"{DES}/helen.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1j4rQeaRPIPU_1VpL7VWDUihfMsXGhAAM/view?usp=sharing"
output = f"{DES}/ibug.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1hB5aas5teU5mKAvww-I_hxRJ_LLgCzjx/view?usp=sharing"
output = f"{DES}/LFPW.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1BKG5OCu6WdyH32km82JpUsDZl1wwU4KS/view?usp=sharing"
output = f"{DES}/WFLW.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/11PbN10yLACDBCjy0ym55-A_J0UP-0Ygt/view?usp=sharing"
output = f"{DES}/300VW.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1_szB20Pvxvskg1xULj8KlggfjdxVbdWf/view?usp=sharing"
output = f"{DES}/AFLW2000_3D.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1YoV7cBpbwsykOQCOS0m5TjeGW4n5YhG8/view?usp=sharing"
output = f"{DES}/helen_image.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)