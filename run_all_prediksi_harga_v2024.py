import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime
import argparse
import mysql.connector as msql
import tensorflow as tf

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hari",type=int, help="Jumlah hari berikutnya yang ingin diprediksi")
    return parser.parse_args()

def get_data_api():
    # Mapping Kode KOMODITI
    # BERAS PREMIUM = 24 Q5
    # CABE MERAH = 2 Q12
    # DAGING AYAM = 4 Q7 
    # TELUR AYAM = 5 Q25
    # DAGIG SAPI = 6 Q6
    # MINYAK GORENG = 7 Q22
    # BAWANG MERAH = Q9
    # BAWANG PUTIH = Q10
    # GULA = Q17
    resp = requests.get('https://kf.kobotoolbox.org/api/v2/assets/avMqa5nFYcGcLBaGiCWk8Z/data.json', headers ={   
            'Authorization': 'Token a3de28eb5ffbea5f319be5202b681ba3c964fb35',
            'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
            })
    resp_dict = resp.json()
    resp_dict=resp_dict['results']

    return resp_dict

def get_input_output_tensors(graph):
    input_tensor = graph.get_tensor_by_name('x:0')
    output_tensor = graph.get_tensor_by_name('sequential_2/dense_2/BiasAdd:0')
    
    return input_tensor, output_tensor

def load_model_ml(komoditas):
    if komoditas == 'Beras Premium':
        with tf.io.gfile.GFile("model_prediksi_beras_premium_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")

        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        print("Model  Beras Premium Loaded")
        kode = 'group_beras/Q5'
        lag = 30
    elif komoditas == 'Beras Medium':
        with tf.io.gfile.GFile("model_prediksi_beras_medium_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Beras Medium Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential_2/dense_2/BiasAdd:0')
        kode = 'group_beras/Q4'
        lag = 30
    elif komoditas == 'Cabai Merah':
        with tf.io.gfile.GFile("model_prediksi_cabai_merah_60.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Cabai Merah Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_sayur/Q12'
        lag = 60
    elif komoditas == 'Cabai Merah Keriting':
        with tf.io.gfile.GFile("model_prediksi_cabai_merah_keriting_90.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Cabai Merah Keriting Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential_3/dense_3/BiasAdd:0')
        kode = 'group_sayur/Q11'
        lag = 90
    elif komoditas == 'Cabai Rawit':
        with tf.io.gfile.GFile("model_prediksi_cabai_rawit_90.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Cabai Rawit Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential_2/dense_2/BiasAdd:0')
        kode = 'group_sayur/Q13'
        lag = 90
    elif komoditas == 'Cabai Rawit Hijau':
        with tf.io.gfile.GFile("model_prediksi_cabai_rawit_hijau_120.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Cabai Rawit Hijau Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential_2/dense_2/BiasAdd:0')
        kode = 'group_sayur/Q14'
        lag = 120
    elif komoditas == 'Minyak Goreng':
        with tf.io.gfile.GFile("model_prediksi_minyak_goreng_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Minyak Goreng Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_keringan/Q22'
        lag = 30
    elif komoditas == 'Daging Ayam':
        with tf.io.gfile.GFile("model_prediksi_daging_ayam_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Daging Ayam Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_ayam/Q7'
        lag = 30
    elif komoditas == 'Daging Sapi':
        with tf.io.gfile.GFile("model_prediksi_daging_sapi_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Daging Sapi Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_daging/Q6'
        lag = 30
    elif komoditas == 'Telur Ayam':
        with tf.io.gfile.GFile("model_prediksi_telur_ayam_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Telur Ayam Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_keringan/Q25'
        lag = 30
    elif komoditas == 'Bawang Merah':
        with tf.io.gfile.GFile("model_prediksi_bawang_merah_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Bawang Merah Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_sayur/Q9'
        lag = 30
    elif komoditas == 'Bawang Putih':
        with tf.io.gfile.GFile("model_prediksi_bawang_putih_30.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Bawang Putih Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_sayur/Q10'
        lag = 30
    elif komoditas == 'Gula Pasir':
        with tf.io.gfile.GFile("model_prediksi_gula_pasir_60.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as model:
            tf.import_graph_def(graph_def, name="")
        print("Model  Gula Pasir Loaded")
        input_tensor = model.get_tensor_by_name('x:0')
        output_tensor = model.get_tensor_by_name('sequential/dense/BiasAdd:0')
        kode = 'group_sayur/Q17'
        lag = 60
    else:
        print('Pilih Komoditas')
    
    return model, lag, kode, input_tensor, output_tensor

def main():
    args = arguments()
    data_all = []
    # LOAD and RUN MODEL
    print("=========Prediksi Harga Bahan Pokok============")
    # ------------------- PROSES data
    resp_dict = get_data_api() # ambil data dari KOBO
    for komoditas in ["Beras Premium","Beras Medium","Cabai Merah","Cabai Merah Keriting","Cabai Rawit","Cabai Rawit Hijau","Minyak Goreng","Daging Ayam","Daging Sapi","Telur Ayam","Bawang Merah","Bawang Putih","Gula Pasir"]:
        model, lag, kode, input_tensor, output_tensor = load_model_ml(komoditas)
        
        list_data = []
        for i in range(len(resp_dict)):
            list_data.append((resp_dict[i].get('Q1'),resp_dict[i].get(kode)))

        awal = (datetime.now() - relativedelta(months=4)).strftime('%Y-%m-%d')
        akhir  = (datetime.now() - relativedelta(days=1)).strftime('%Y-%m-%d')
        df = pd.DataFrame(list_data, columns = ["Tanggal",'Harga'])
        df = df[(df['Tanggal']>=awal) & (df['Tanggal']<=akhir)]

        df['Tanggal'] = pd.to_datetime(df['Tanggal'],  format='%Y-%m-%d') 
        #df = df.set_index('Tanggal').resample('1D').median()
        df = df[df['Harga'] != '0']
        print(df)
        df = df.groupby(['Tanggal']).median()
        idx = pd.date_range(df.index.min(),df.index.max())

        df.index = pd.DatetimeIndex(df.index)

        s = df.reindex(idx, method='nearest')

        data = s.reset_index()
        data.columns = ["Tanggal", "Harga"]
        df = data[['Tanggal','Harga']].dropna()
        df['Tanggal'] = pd.to_datetime(df['Tanggal'],  format='%Y-%m-%d')
        data['Harga']=data['Harga'].astype(int)

        max_price = data.iloc[:,1:2].values.max()
        min_price = data.iloc[:,1:2].values.min()

        # cek outlier
        while max_price > data['Harga'].mean()*3:
            data['Harga'][data['Harga']==max_price]=data['Harga'].mean()
            max_price = data.iloc[:,1:2].values.max()
        while min_price < int(data['Harga'].mean()/3):
            data['Harga'][data['Harga']==min_price]=data['Harga'].mean()
            min_price = data.iloc[:,1:2].values.min()

        sc = MinMaxScaler(feature_range=(0,1))

        hari = args.hari
        
        data_baru = data.copy()
        X_test = []
        pred = []
        today = datetime.now().strftime('%Y-%m-%d')
        th = np.arange(5000,300000,50)
        for i in range(hari):
            test_set_scaled = sc.fit_transform(data_baru.iloc[len(data_baru)-lag:,1:2].values)
            X_test = [test_set_scaled[:, 0]]
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            tgl = (data_baru['Tanggal'].iloc[-1] + relativedelta(days=1)).strftime('%Y-%m-%d')

            # Buat Prediksi
            with tf.compat.v1.Session(graph=model) as sess:
                hasil_pred = sess.run(output_tensor, feed_dict={input_tensor: X_test})
                hasil_pred = sc.inverse_transform(hasil_pred)
            selisih = abs(hasil_pred[0][0]-th)
            hasil_pred = th[list(selisih).index(min(selisih))]
            pred.append((today, komoditas, args.hari, tgl, hasil_pred))

            #update data_baru
            data_baru.loc[len(data_baru)] = [tgl,hasil_pred]
            data_baru['Tanggal'] = pd.to_datetime(data_baru['Tanggal'],  format='%Y-%m-%d')

            data_all.append((today, komoditas, args.hari, tgl, int(hasil_pred)))
        

    df_data_all = pd.DataFrame(data_all,columns= ["updated_at", "commodity", "step", "pred_date","price"])
    print(df_data_all)



    # ---------------------------------------------------------------------------------------------------------------------------
    # SAVE TO DB
    # KONEKSI KE DB
    try:
        conn = msql.connect(host='103.108.190.91', user='opendata',  
                            password='Dataconf2023!', database='db_data_bdg')
        print(".....Berhasil Terhubung ke Database....")
    except Exception as e:
        print("Alert !! ....  Tidak dapat terhubung dengan Database .... ")
    
    # CEK DATA EXISTING
    cursor = conn.cursor(buffered=True)

    # EXECUTE
    cursor.execute("DELETE FROM prediksi_harga WHERE step=%s",(hari,))
    for ii in range(len(df_data_all)):
        print((datetime.strptime(df_data_all.iloc[ii][0],"%Y-%m-%d").date(),df_data_all.iloc[ii][1],df_data_all.iloc[ii][2]))

        sql_insert = """INSERT INTO prediksi_harga (updated_at, commodity, step, pred_date, price) VALUES (%s,%s,%s,%s,%s)"""
        cursor.execute(sql_insert,data_all[ii])
        conn.commit()
        print("-----  Data berhasil di Insert  --------")

    conn.close()

if __name__ == "__main__":
    main()