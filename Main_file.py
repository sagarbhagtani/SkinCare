from acne.Model_3 import initiateAcne as ac
from Application.DarkCircles.DarkCircles import initiateDarkCircle as dc
from Application.Wrinkles.Wrinkles import initiateWrinkles as wr
from Application.Blackheads.Blackheads import initiateBlackheads as bh
from Application.OilySkin.OilySkin import initiateOilySkin as os
from flask import request, jsonify
from flask import Flask

app = Flask(__name__)
# CORS(app)
# # Restrict CORS to your React app's origin
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route('/techsharthi', methods=['POST'])
def fullname():
    data = request.json
    id = data.get('id', '')
    imagepathCenter = data.get('imagepathCenter', '')
    imagepathLeft = data.get('imagepathLeft', '')
    imagepathRight = data.get('imagepathRight', '')
    imageNose = data.get('imageNose', '')
    ImageLocations={'imagepathCenter':imagepathCenter,'imagepathLeft':imagepathLeft,'imagepathRight':imagepathRight,'imageNose':imageNose}
    print(ImageLocations)

    print(imagepathCenter)
    print(imagepathLeft)
    print(imagepathRight)
    print(imageNose)

    print(id)

    imagepath = "C:/Users/ADMIN"
    #id = "1112_121448"
    ResponseList = []
    AcneRes = ac(ImageLocations, id)
    DC_Res = dc(ImageLocations, id)
    WR_Res = wr(ImageLocations, id)
    BH_Res = bh(ImageLocations, id)
    OS_Res=os(ImageLocations, id)

    ResponseList.append(AcneRes)
    ResponseList.append(BH_Res)
    ResponseList.append(WR_Res)
    ResponseList.append(DC_Res)
    ResponseList.append(OS_Res)
    print(ResponseList)
    #return jsonify({'full_name': full_name})
    return jsonify(ResponseList)

if __name__ == '__main__':
    app.run(debug=True)







