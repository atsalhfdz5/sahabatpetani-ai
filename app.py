import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__, 
            template_folder='page', 
            static_folder='.', 
            static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model YOLO (best.pt)
path_model = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(path_model)

print("Model yang digunakan:", path_model)
print("Nama kelas:", model.names)

# Setelan akurasi pengujian (40%)
THRESHOLD_AKURASI = 0.25



def preprocess_image(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l=clahe.apply(l)
    img=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    h,w=img.shape[:2]
    s=640/max(h,w)
    if s<1:
        img=cv2.resize(img,(int(w*s),int(h*s)))
    return img

# Pemetaan deskripsi otomatis berdasarkan kata kunci yang terdeteksi model
def dapatkan_deskripsi(label_raw):
    label_lower = str(label_raw).lower()
    if 'bacterial' in label_lower or 'blight' in label_lower:
        return 'Penyakit Hawar Daun Bakteri. Gejala berupa garis kemerahan atau kecoklatan pada tepi daun yang lambat laun menyebar hingga daun layu dan kering.'
    elif 'brown' in label_lower or 'spot' in label_lower:
        return 'Penyakit Bercak Coklat. Gejala berupa bercak berbentuk oval berwarna coklat tua pada permukaan daun yang dapat mengurangi proses fotosintesis tanaman.'
    elif 'blast' in label_lower:
        return 'Penyakit Blas Daun. Gejala berupa bercak berbentuk belah ketupat dengan pusat berwarna abu-abu, memicu kerusakan parah pada daun padi muda.'
    else:
        return f'Terdeteksi gejala {label_raw}. Lakukan pemantauan berkala dan pemberian nutrisi tepat pada tanaman padi.'

def core_proses_ai(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_asli = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_asli is None:
            return {'terdeteksi': False, 'label': 'Gambar Korup', 'score': 0, 'deskripsi': '', 'img_output': None}

        img_asli=preprocess_image(img_asli)
        results=model(img_asli, conf=THRESHOLD_AKURASI, iou=0.45)[0]
        if len(results.boxes)==0:
            results=model(preprocess_image(cv2.GaussianBlur(img_asli,(3,3),0)), conf=0.15, iou=0.45)[0]
        
        terdeteksi_valid = False
        label_tertinggi = "Tidak Terdeteksi"
        score_tertinggi = 0.0

        # Pengaman tambahan untuk realtime: jika boks mendeteksi terlalu banyak objek acak (> 5 boks)
        if len(results.boxes) > 5:
            return {'terdeteksi': False, 'label': 'Tidak Terdeteksi', 'score': 0, 'deskripsi': 'Sistem tidak mendeteksi adanya gejala penyakit tanaman padi pada foto ini.', 'img_output': img_asli}

        for box in results.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_idx = int(box.cls[0])
            
            pred_label = model.names[class_idx]

            terdeteksi_valid = True
            if confidence > score_tertinggi:
                score_tertinggi = confidence
                label_tertinggi = pred_label
            
            cv2.rectangle(img_asli, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            teks = f"{pred_label} {confidence:.2f}"
            cv2.putText(img_asli, teks, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        return {
            'terdeteksi': terdeteksi_valid,
            'label': label_tertinggi,
            'score': score_tertinggi,
            'deskripsi': dapatkan_deskripsi(label_tertinggi) if terdeteksi_valid else "Sistem tidak mendeteksi adanya gejala penyakit tanaman padi pada foto ini.",
            'img_output': img_asli
        }
    except Exception as e:
        print(f"Error AI processing: {e}")
        return {'terdeteksi': False, 'label': 'Error', 'score': 0, 'deskripsi': 'Gagal memproses analisis.', 'img_output': None}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    img=preprocess_image(img)
    img_output=img.copy()

    # Confidence disarankan 0.30 - 0.40
    # Single inference
    results = model(img, conf=0.25, iou=0.45)[0]

    print("Jumlah box:", len(results.boxes))

    for box in results.boxes:
        print(
        "Label:", model.names[int(box.cls[0])],
        "Confidence:", float(box.conf[0])
    )

    info_penyakit = {
        'Bacterial Leaf Blight': {
            'nama': 'Hawar Daun Bakteri (Bacterial Leaf Blight)',
            'desc': 'Penyakit yang disebabkan oleh bakteri Xanthomonas oryzae. Gejalanya berupa garis memanjang kecokelatan pada tepi daun.',
            'solusi': 'Gunakan bakterisida, hindari pemberian nitrogen berlebihan, dan lakukan sanitasi lahan.'
        },
        'Brown Spot': {
            'nama': 'Bercak Cokelat (Brown Spot)',
            'desc': 'Disebabkan oleh jamur Helminthosporium oryzae.',
            'solusi': 'Gunakan fungisida dan lakukan pemupukan berimbang.'
        },
        'Leaf Blast': {
            'nama': 'Blas Daun (Leaf Blast)',
            'desc': 'Disebabkan oleh jamur Magnaporthe oryzae.',
            'solusi': 'Gunakan varietas tahan dan fungisida sistemik.'
        }
    }

    if len(results.boxes) == 0:
        os.makedirs("static", exist_ok=True)
        output_path = "static/hasil_prediksi.jpg"
        cv2.imwrite(output_path, img_output)

        return jsonify({
            'result_image_url': '/' + output_path,
            'terdeteksi': False,
            'nama_penyakit': 'Tidak Terdeteksi',
            'confidence': '-',
            'deskripsi': 'Sistem tidak mendeteksi gejala penyakit pada daun padi.',
            'solusi': 'Pastikan gambar merupakan daun padi dan memiliki kualitas yang baik.'
        })

    # Ambil prediksi dengan confidence tertinggi
    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))

    conf = float(best_box.conf[0])
    cls_id = int(best_box.cls[0])
    label = model.names[cls_id]

    print("Label :", label)
    print("Confidence :", conf)

    # Confidence minimal
    if conf < 0.50:
        os.makedirs("static", exist_ok=True)
        output_path = "static/hasil_prediksi.jpg"
        cv2.imwrite(output_path, img_output)

        return jsonify({
            'result_image_url': '/' + output_path,
            'terdeteksi': False,
            'nama_penyakit': 'Tidak Terdeteksi',
            'confidence': '-',
            'deskripsi': 'Confidence model terlalu rendah sehingga hasil diabaikan.',
            'solusi': 'Coba ambil foto lebih jelas dan fokus pada daun.'
        })

    if label not in info_penyakit:
        os.makedirs("static", exist_ok=True)
        output_path = "static/hasil_prediksi.jpg"
        cv2.imwrite(output_path, img_output)

        return jsonify({
            'result_image_url': '/' + output_path,
            'terdeteksi': False,
            'nama_penyakit': 'Tidak Dikenal',
            'confidence': '-',
            'deskripsi': 'Objek yang terdeteksi bukan termasuk kelas penyakit yang dikenali.',
            'solusi': 'Gunakan gambar daun padi yang sesuai.'
        })

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.putText(
        img_output,
        f"{label} ({conf*100:.1f}%)",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    os.makedirs("static", exist_ok=True)
    output_path = "static/hasil_prediksi.jpg"
    cv2.imwrite(output_path, img_output)

    return jsonify({
        'result_image_url': '/' + output_path,
        'terdeteksi': True,
        'nama_penyakit': info_penyakit[label]['nama'],
        'confidence': f"{conf*100:.1f}%",
        'deskripsi': info_penyakit[label]['desc'],
        'solusi': info_penyakit[label]['solusi']
    })

@socketio.on('video_frame')
def handle_video_frame(data_url):
    try:
        header, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        hasil = core_proses_ai(image_bytes)
        
        if hasil['img_output'] is not None:
            _, buffer = cv2.imencode('.jpg', hasil['img_output'])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            response_url = f"data:image/jpeg;base64,{jpg_as_text}"
            
            emit('response_frame', {
                'image_url': response_url,
                'terdeteksi': hasil['terdeteksi'],
                'label': hasil['label'],
                'confidence': f"{hasil['score'] * 100:.1f}%" if hasil['terdeteksi'] else "-",
                'deskripsi': hasil['deskripsi']
            })
    except Exception as e:
        print(f"Socket error: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=False)