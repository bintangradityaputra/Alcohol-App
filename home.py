import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import pandas as pd
import io 
from awesome_table import AwesomeTable
from awesome_table.column import Column
from sklearn.preprocessing import MinMaxScaler


with st.container():
    with st.sidebar:
        choose = option_menu("Menu Bar", ["Home", "Dataset", "Pre-Processing", "Modelling", "Implementasi"],
                            icons=['house', 'clipboard-data', 'activity', 'pie-chart', 'gear'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#f5bc00"},
            "icon": {"color": "black", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee", "color" : "black"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
        )

        
    if choose == "Home":
        st.title("Alcohol Effect on Study")
        st.subheader("Pengertian Alcohol")
        st.markdown('<div style="text-align: justify;">Alcohol adalah segala minuman difermentasi yang mengandung etil alkohol atau etanol sebagai zat yang memabukkan. Biasanya minuman alkohol dibuat dari gula yang difermentasi dalam buah-buahan, seperti beri, biji-bijian, dan bahan-bahan lain seperti getah tanaman, umbi-umbian, madu, dan susu. Fermentasi berbagai bahan ini dapat menghasilkan cairan yang memiliki kadar alkohol yang lebih besar dan lebih kuat.</div>', unsafe_allow_html=True)
        st.image('alcohol.png')
        st.markdown('<div style="text-align: justify;">Secara umum, jenis minuman keras dibedakan menjadi beberapa macam. Pertama adalah bir yaitu minuman alkohol yang terbuat dari bahan malt seperti jagung, beras, dan hop. Biasanya bir memiliki kandungan alkohol berkisar antara 2 persen hingga 8 persen. Kedua adalah anggur, yaitu minuman alkohol yang terbuat dari fermentasi jus anggur atau buah-buahan lain seperti apel, ceri, beri, atau prem. Pembuatan anggur dimulai dengan panen buah , yang sarinya difermentasi dalam tong besar di bawah kontrol suhu yang ketat. Saat fermentasi selesai, campuran disaring, didiamkan, dan dibotolkan.</div>', unsafe_allow_html=True)
        
    elif choose == "Dataset":
        st.title("Dataset Alcohol Effect On Study")
        st.subheader("About Dataset")
        st.markdown('<div style="text-align: justify;"> Dataset ini diperoleh dari pencapaian siswa dalam sekolah menengah di dua sekolah yang terletak di Portugis. Atribut dari dataset ini sendiri meliput nilai siswa, demografi, fitur sosial dan terkait sekolah. Dataset ini dikumpulkan dengan menggunakan laporan sekolah dan juga dari kuesioner.</div>', unsafe_allow_html=True)

        df = pd.read_csv("https://raw.githubusercontent.com/bintangradityaputra/contoh/master/Maths.csv")
        df

        st.subheader("Feature Description")

        sample_data = [
            {'fitur' : "school", "deskripsi" : "Fitur school menjelaskan dimana murid bersekolah (tedapat dua sekolah dengan code 'GP' - Gabriel Pereira atau 'MS' - Mousinho da Silveira)" },
            {'fitur' : "sex", "deskripsi" : "Fitur sex merupakan jenis kelamin dari murid (code : 'F' - perempuan atau 'M' - laki-laki)" },
            {'fitur' : "age", "deskripsi" : "Fitur age merupakan umur dari murid (parameter umur dari 15 sampai 22)" },
            {'fitur' : "address", "deskripsi" : "Fitur address merupakan alamat dimana murid tinggal (parameter alamat di lambangkan dengan 'U' - urban atau 'R' - rural)" },
            {'fitur' : "famsize", "deskripsi" : "Fitur famsize merupakan jumlah anggota keluarga yang dimiliki oleh murid (parameter : 'LE3' - jumlah anggota <= 3 atau 'GT3' - jumlah anggota keluarga > 3)" },
            {'fitur' : "Pstatus", "deskripsi" : "Fitur Pstatus menjelaskan apakah murid masih tinggal bersama orang tuanya atau tidak (parameter 'T' - living together(tinggal bersama orang tua) atau 'A' - apart(tinggal di apartement)" },
            {'fitur' : "Medu", "deskripsi" : "Fitur Medu berarti mother education atau pendidikan ibu (parameter untuk fitur Medu ini 0(none), 1-(pendidikan dasar kelas 4 sekolah dasar), 2-(kelas 5 sampai kelas 9), 3-(pendidikan menengah), 4-(pendidikan tinggi)" },
            {'fitur' : "Fedu", "deskripsi" : "Fitur Fedu berarti father education atau pendidikan ibu (parameter untuk fitur Medu ini 0(none), 1-(pendidikan dasar kelas 4 sekolahb dasar), 2-(kelas 5 sampai kelas 9), 3-(pendidikan menengah), 4-(pendidikan tinggi)" },
            {'fitur' : "Mjob", "deskripsi" : "Fitur Mjob berarti jenis pekerjaan yang dimiliki oleh ibu dari murid (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')" },
            {'fitur' : "Fjob", "deskripsi" : "Fitur Fjob berarti jenis pekerjaan yang dimiliki oleh ayah dari murid (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')" },
            {'fitur' : "reason", "deskripsi" : "Fitur reason merupakan alasan mengapa murid memilih masuk kedalam sekolah tersebut (nominal: close to 'home', school 'reputation', 'course' preference or 'other')" },
            {'fitur' : "guardian", "deskripsi" : "Fitur guardian menjelaskan siapa yang meelindungi murid (nominal: 'mother', 'father' or 'other')" },
            {'fitur' : "traveltime", "deskripsi" : "Fitur traveltime berarti berapa lama waktu yang dibutuhkan untuk murid pergi ke sekolah (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)" },
            {'fitur' : "studytime", "deskripsi" : "studytime berarti berapa lama waktu total murid belajar dalam waktu 1 minggu (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)" },
            {'fitur' : "failures", "deskripsi" : "Fitur failures merupakan jumlah kegagalan kelas yang pernah di tempuh oleh murid (numeric: n if 1<=n<3, else 4)" },
            {'fitur' : "schoolsup", "deskripsi" : "Fitur shoolsup menandakan apakah murid memiliki dukungan pendidikanm tambahan atau tidak (binary: yes or no)" },
            {'fitur' : "famsup", "deskripsi" : "Fitur famsup menandakan apakah murid memiliki dukungan keluarga dalam pendidikan atau tidak (binary: yes or no)" },
            {'fitur' : "paid", "deskripsi" : "Fitur paid menandakan apakah murid melakukan kelas berbayar dalam mata pelajaran (Math or Portuguese) atau tidak (binary: yes or no)" },
            {'fitur' : "activity", "deskripsi" : "Fitur activity menandakan apakah seorang murid mengikuti kegiatan ekstrakurikuler atau tidak saat disekolah (binary: yes or no)" },
            {'fitur' : "nursery", "deskripsi" : "Fitur nursery menandakan apakah murid menempuh pendidikan di taman kanak-kanak atau tidak (binary: yes or no)" },
            {'fitur' : "higher", "deskripsi" : "Fitur higher menandakan pakah murid ingin menempuh pendidikan yang lebih tinggi atau tidak (binary: yes or no)" },
            {'fitur' : "internet", "deskripsi" : "Fitur internet menandakan apakah dirumah murid terdapat akses internet atau tidak (binary: yes or no)" },
            {'fitur' : "romantic", "deskripsi" : "Fitur romantic adalah apakah murid memiliki hubungan dengan pasangan atau tidak (binary: yes or no)" },
            {'fitur' : "famrel", "deskripsi" : "Fitur famrel merupakan seberapa berkualitas hubungan murid dengan keluarganya (numeric: 1 & 2- (sangat rendah), 3 - (rendah), 4 - (tinggi), 5 - (sangat tinggi))" },
            {'fitur' : "freetime", "deskripsi" : "Fitur freetime merupakan seberapa banyak waktu luang yang dimiliki murid setelah pulang sekolah (numeric: 1 & 2- (sangat rendah), 3 - (rendah), 4 - (tinggi), 5 - (sangat tinggi))" },
            {'fitur' : "goout", "deskripsi" : "Fitur goout merupakan seberapa sering murid keluar atau pergi bersama temannya (numeric: 1 & 2- (sangat rendah), 3 - (rendah), 4 - (tinggi), 5 - (sangat tinggi))" },
            {'fitur' : "Dalc", "deskripsi" : "Fitur Dalc merupakan berapa jumlah alkohol yang dikonsumsi oleh murid pada hari kerja (numeric: 1 & 2- (sangat rendah), 3 - (rendah), 4 - (tinggi), 5 - (sangat tinggi))" },
            {'fitur' : "Walc", "deskripsi" : "Fitur Walc merupakan berapa jumlah alkohol yang dikonsumsi oleh murid pada hari libur (numeric: 1 & 2- (sangat rendah), 3 - (rendah), 4 - (tinggi), 5 - (sangat tinggi))" },
            {'fitur' : "health", "deskripsi" : "Fitur health menandakan tingkat kesehatan para murid (numeric: 1 & 2 - (menandakan status kesehatan yang sangat buruk), 3 - (menandakan status kesehatan yang buruk), 4 - (menandakan status kesehatan yang bagus), 5 - (menandakan tingkat kesehatan yang sangat bagus))" },
            {'fitur' : "absences", "deskripsi" : "Fitur absences merupakan nomor absen yang dimiliki murid saat disekolah  (numeric: dari 0 sampai 93)" },
        ]

        AwesomeTable(pd.json_normalize(sample_data), columns=[
            Column(name='fitur', label='Fitur'),
            Column(name='deskripsi', label='Deskripsi'),
        ], key="deskripsi")


    elif choose ==  "Pre-Processing" :
        st.title ("Pre-Processing")
        st.subheader("Pengertian Pre-Processing")
        st.markdown('<div style="text-align: justify;">Data preprocessing adalah teknik awal data mining untuk mengubah raw data (data mentah) menjadi format dan informasi yang lebih efisien dan bermanfaat. Format pada raw data yang diambil dari berbagai macam sumber seringkali mengalami error, missing value, dan tidak konsisten. Sehingga, perlu dilakukan pembenahan format agar hasil data mining tepat dan akurat.Preprocessing melibatkan validasi dan imputasi data, dimana validasi ini bertujuan untuk menilai tingkat kelengkapan dan akurasi data. Sementara imputasi data bertujuan untuk memperbaiki kesalahan dan memasukkan missing value, melalui program business process automation (BPA).</div>', unsafe_allow_html=True)

        # dataset
        st.subheader("Data sebelum dilakukan pre-processing")
        df = pd.read_csv("https://raw.githubusercontent.com/bintangradityaputra/contoh/master/Maths.csv")
        df

        # seleksi fitur
        st.subheader("Seleksi fitur")
        df = df[["sex", "age", "Pstatus", "Medu", "Fedu", "studytime", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]]
        df

        # mengubah data categorial menjadi numeric
        st.subheader("Mengubah fitur sex dari categorical menjadi numeric")
        sex = pd.get_dummies(df.sex)
        df = df.drop(columns = ['sex'])
        df[["F", "M"]] = sex
        df
        st.subheader("Mengubah fitur Pstatus dari categorical menjadi numeric")
        Pstatus = pd.get_dummies(df.Pstatus)
        df = df.drop(columns = ['Pstatus'])
        df[['A', 'T']] = Pstatus
        df

        # drop label
        st.subheader("Data tanpa label")
        X = df.drop(columns = "health")
        y = df.health
        X

        # Normalisasi
        st.subheader("Pengertian Normalisasi")
        st.markdown('<div style="text-align: justify;">Normalisasi merupakan sebuah teknik dalam logical desain sebuah basis data yang mengelompokkan atribut dari suatu relasi sehingga membentuk struktur relasi yang baik (tanpa redudansi). Normalisasi adalah proses pembentukan struktur basis data sehingga sebagian besar ambiguity bisa dihilangkan.</div>', unsafe_allow_html=True)
        st.subheader("Normalisasi menggunakan Min-Max Scaler")
        st.markdown('<div style="text-align: justify;">Normalisasi min-max biasanya memungkinkan untuk mengubah data dengan skala yang bervariasi sehingga tidak ada dimensi tertentu yang mendominasi statistik, dan tidak perlu membuat asumsi kuat tentang distribusi data.</div>', unsafe_allow_html=True)
        st.text("Rumus Min-Max Scaler :")
        st.latex(r'''x^{'} = \frac{x - x_{min}}{x_{max}-x_{min}}''')
        # Normalisasi data
        st.subheader("Data setelah di normalisasi")
        scaler = MinMaxScaler()
        scaler.fit(X[['age']])
        X[['age']] = scaler.transform(X[['age']])
        X

    elif choose == "Modelling" :
        st.title("Modelling")
        df = pd.read_csv("https://raw.githubusercontent.com/bintangradityaputra/contoh/master/Maths.csv")
        # seleksi fitur
        df = df[["sex", "age", "Pstatus", "Medu", "Fedu", "studytime", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]]

        # mengubah data categorial menjadi numeric
        sex = pd.get_dummies(df.sex)
        df = df.drop(columns = ['sex'])
        df[["F", "M"]] = sex

        Pstatus = pd.get_dummies(df.Pstatus)
        df = df.drop(columns = ['Pstatus'])
        df[['A', 'T']] = Pstatus

        # label
        X = df.drop(columns = "health")
        y = df.health

        # st.subheader("Data setelah di normalisasi")
        scaler = MinMaxScaler()
        scaler.fit(X[['age']])
        X[['age']] = scaler.transform(X[['age']])

        st.subheader("Split Dataset")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
        st.markdown('<div style="text-align: justify;">Dari 395 data yang ada pada dataset, dilakukan splitting atau pembagian data menjadi data latih dan data uji. Sehingga diperoleh 316 data latih dan 79 data uji.</div>', unsafe_allow_html=True)


        # KNN Model
        st.subheader("K-Nearest Neighbor")
        st.markdown('<div style="text-align: justify;">K-NN adalah algoritma klasifikasi yang bekerja dengan mengambil sejumlah K data terdekat (tetangganya) sebagai acuan untuk menentukan kelas dari data baru. Algoritma ini mengklasifikasikan data berdasarkan similarity atau kemiripan atau kedekatannya terhadap data lainnya.</div>', unsafe_allow_html=True)
        st.write("Rumus :")
        st.latex(r'''d(x,y) = \sqrt{\sum_{i=1}^{n}(x-y)^{2}}''')
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors = 3)
        # Fit the classifier to the data
        knn.fit(X_train, y_train)
        # Akurasi Score
        knn.score(X_test, y_test)
        st.write("Akurasi Skor dengan menggunakan model K-NN :", round(knn.score(X_test, y_test)*100), 2,'%')

        # Gaussian Naive Baiyes Model
        st.subheader("Gaussian Naive Bayes")
        st.markdown('<div style="text-align: justify;">Naïve Bayes Classifier merupakan sebuah metoda klasifikasi yang berakar pada teorema Bayes . Metode pengklasifikasian dg menggunakan metode probabilitas dan statistik yg dikemukakan oleh ilmuwan Inggris Thomas Bayes , yaitu memprediksi peluang di masa depan berdasarkan pengalaman di masa sebelumnya sehingga dikenal sebagai Teorema Bayes . Ciri utama dr Naïve Bayes Classifier ini adalah asumsi yg sangat kuat (naïf) akan independensi dari masing-masing kondisi / kejadian.</div>', unsafe_allow_html=True)
        st.text("Rumus : ")
        st.latex(r'''P(C_{k}|x) = \frac{P(C_{k})P(x|C_{k})}{P(x)}''')
        # akurasi
        from sklearn.naive_bayes import GaussianNB
        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)
        gaussian.score(X_test, y_test)
        st.write("Akurasi Skor dengan menggunakan model Gaussian Naive Bayes:", round(gaussian.score(X_test, y_test)*100), 2,'%')

        # Decision Tree Model
        st.subheader("Decision Tree")
        st.markdown('<div style="text-align: justify;">Decision tree adalah jenis algoritma klasifikasi yang strukturnya mirip seperti sebuah pohon yang memiliki akar, ranting, dan daun. Simpul akar (internal node) mewakili fitur pada dataset, simpul ranting (branch node) mewakili aturan keputusan (decision rule), dan tiap-tiap simpul daun (leaf node) mewakili hasil keluaran.</div>', unsafe_allow_html=True)
        st.write("Rumus Entropy :")
        st.latex(r'''Entropy (S) = \sum_{i=1}^{n}-\pi * log_{2}\pi ''')
        from sklearn import tree
        tree = tree.DecisionTreeClassifier(criterion="gini")
        tree = tree.fit(X, y)
        tree.score(X_test, y_test)
        st.write("Akurasi Skor dengan menggunakan model Decision Tree:", round(tree.score(X_test, y_test)*100), 2,'%')

    elif choose == "Implementasi" :
        st.title("PREDIKSI PENGARUH ALKOHOL DI DALAM PEMBELAJARAN")
        # code 
        df = pd.read_csv("https://raw.githubusercontent.com/bintangradityaputra/contoh/master/Maths.csv")
        # seleksi fitur
        df = df[["sex", "age", "Pstatus", "Medu", "Fedu", "studytime", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]]

        # mengubah data categorial menjadi numeric
        sex = pd.get_dummies(df.sex)
        df = df.drop(columns = ['sex'])
        df[["F", "M"]] = sex

        Pstatus = pd.get_dummies(df.Pstatus)
        df = df.drop(columns = ['Pstatus'])
        df[['A', 'T']] = Pstatus

        # label
        X = df.drop(columns = "health")
        y = df.health

        scaler = MinMaxScaler()
        norm = X[['age']]
        scaled = scaler.fit_transform(norm)
        features_names = norm.columns.copy()
        scaled_features = pd.DataFrame(scaled, columns = features_names)
        X[['age']] = scaled

        # st.subheader("Split Dataset")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # X_train.shape, X_test.shape, y_train.shape, y_test.shape
        # end code

        age = st.number_input('Masukkan Umur (15-22) :')
        sex = st.radio(
        "Pilih Jenis Kelamin :",
        ('Laki-Laki (M)', 'Perempuan (F)'))
        Pstatus = st.radio(
        "Pilih Status :",
        ('Living Together (T)', 'Apart (A)'))
        Medu = st.radio(
        "Pilih Penididkan Ibu :",
        ('0 - (none)', '1 - (Kelas 4 Dasar)', '2 - (Kelas 5 - 9)', '3 - (Pendidikan Menengah)', '4 - (Pendidikan Tinggi)'))
        Fedu = st.radio(
        "Pilih Pendidkan Ayah :",
        ('0 - (none)', '1 - (Kelas 4 Dasar)', '2 - (Kelas 5 - 9)', '3 - (Pendidikan Menengah)', '4 - (Pendidikan Tinggi)'))
        studytime = st.radio(
        "Waktu belajar dalam seminggu :",
        ('1 - (< 2 jam)', '2 - (2 - 5 jam)', '3 - (5 - 10 jam)', '4 - (> 10 jam)'))
        famrel = st.radio(
        "Kualitas Hubungan dengan Keluarga :",
        ('1 & 2 - (sangat rendah)', '3 - (rendah)', '4 - (tinggi)', '5 - (sangat tinggi)'))
        freetime = st.radio(
        "Waktu Luang :",
        ('1 & 2 - (sangat rendah)', '3 - (rendah)', '4 - (tinggi)', '5 - (sangat tinggi)'))
        goout = st.radio(
        "Waktu Pergi bersama Teman :",
        ('1 & 2 - (sangat rendah)', '3 - (rendah)', '4 - (tinggi)', '5 - (sangat tinggi)'))
        Dalc = st.radio(
        "Konsumsi Alcohol ketika Hari Kerja :",
        ('1 & 2 - (sangat rendah)', '3 - (rendah)', '4 - (tinggi)', '5 - (sangat tinggi)'))
        Walc = st.radio(
        "Konsumsi Alcohol ketika Hari Libur :",
        ('1 & 2 - (sangat rendah)', '3 - (rendah)', '4 - (tinggi)', '5 - (sangat tinggi)'))

        # normalisasi inputan
        predict = [age]
        fitur = [sex, Pstatus, Medu, Fedu, studytime, famrel, freetime, goout, Dalc, Walc]
        norm_max = X.max()
        norm_min = X.min()
        pred = ((predict - norm_min)/norm_max - norm_min)
        pred = np.array(pred).reshape(1, -1)

        # choose model
        choose_model = st.selectbox("Pilih Model :", ('NONE', 'KNN', 'NAIVE BAYES', 'DECISION TREE'))

        if choose_model == "KNN" :        
            submit = st.button("Predict Score")
            if submit :
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score
                # Create KNN classifier
                knn = KNeighborsClassifier(n_neighbors = 3)
                # Fit the classifier to the data
                knn.fit(X_train, y_train)
                # Akurasi Score
                knn.score(X_test, y_test)
                knn_pred = knn.predict(X_test)
                predict_knn = knn.predict(pred)[0]
                st.write("Hasil Prediksi = ", predict_knn)

        elif choose_model == "NAIVE BAYES" :
            submit = st.button("Predict Score")
            if submit :
                from sklearn.naive_bayes import GaussianNB
                gaussian = GaussianNB()
                gaussian.fit(X_train, y_train)
                gaussian.score(X_test, y_test)
                predict_gaussian = gaussian.predict(X_test)
                predict_gaussian = gaussian.predict(pred)[0]
                st.write("Hasil Prediksi = ", predict_gaussian)

        elif choose_model == "DECISION TREE" :
            submit = st.button("Predict Score")
            if submit :
                from sklearn import tree
                tree = tree.DecisionTreeClassifier(criterion="gini")
                tree = tree.fit(X, y)
                tree.score(X_test, y_test)
                predict_tree = tree.predict(X_test)
                predict_tree = tree.predict(pred)[0]
                st.write("Hasil Prediksi = ", predict_tree)









