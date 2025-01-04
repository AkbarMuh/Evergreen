import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model with generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
        Bertindaklah sebagai asisten pertanian virtual. Anda dapat membantu petani dengan pertanyaan tentang pertanian, tanah, cuaca, dan hasil panen.
        Saya Evergreen Assistant, Saya adalah asisten pertanian berbasis AI yang dirancang untuk membantu petani cabai rawit dalam mengelola lahan mereka secara efisien. Berikut adalah hal-hal yang dapat saya bantu:

        Memberikan informasi terkait kondisi tanah, seperti kelembaban, pH, dan kandungan nutrisi (Nitrogen, Fosfor, Kalium).
        Menganalisis kondisi cuaca berdasarkan prakiraan terbaru, termasuk mitigasi risiko penyakit terkait cuaca.
        Membantu memantau kesehatan tanaman berdasarkan prediksi penyakit seperti patek, antraknosa, dan penyakit lainnya.
        Memberikan panduan tentang perawatan tanaman, termasuk irigasi, pemupukan, dan pengendalian hama.
        Data terbaru yang tersedia dari lapangan:

        Suhu rata-rata: {Suhu dari data terakhir} °C
        Kelembaban: {Kelembaban dari data terakhir} %
        pH tanah: {pH tanah dari data terakhir}
        Kandungan Nutrisi: Nitrogen {Nitrogen dari data terakhir} ppm, Fosfor {Fosfor dari data terakhir} ppm, Kalium {Kalium dari data terakhir} ppm
        Cuaca: {Kondisi cuaca dari data prakiraan terbaru}
        Prediksi Penyakit: {Prediksi penyakit dari data terbaru}
        Saya siap membantu menjawab pertanyaan Anda tentang pertanian, baik terkait kondisi lapangan, cuaca, maupun tindakan yang perlu dilakukan untuk meningkatkan hasil panen.
        
        """
)

# Initialize chat session
chat_session = model.start_chat(
    history=[
        {
            "role": "model",
            "parts": [
                "Halo! 👋 Saya adalah asisten pertanian virtual Anda. 😊\n"
                "Apa yang bisa saya bantu untuk meningkatkan produktivitas pertanian Anda? 🌱\n"
            ],
        }
    ]
)

# Load datasets
conditions_df = pd.read_csv('Data_Kondisi_Pertanian.csv')
forecast_df = pd.read_csv('data_prakiraan_cuaca_mitigasi_penyakit.csv')
predictions_df = pd.read_csv('data_prediksiPenyakit_pertumbuhan_deteksiHama_ESP32-CAM.csv')

# Streamlit App
st.title("Dashboard Kondisi Pertanian")
st.markdown("## Sistem Pemantauan dan Analisis Data Pertanian")

# Sidebar options
data_option = st.sidebar.selectbox("Pilih Data untuk Ditampilkan:", ["Kondisi Pertanian", "Prakiraan Cuaca", "Prediksi Penyakit", "Chatbot Pertanian"])

if data_option == "Kondisi Pertanian":
    st.header("Data Kondisi Pertanian")

    # Filter options
    st.sidebar.subheader("Filter Data")
    start_date = st.sidebar.date_input("Mulai Tanggal", pd.to_datetime(conditions_df['Timestamp']).min())
    end_date = st.sidebar.date_input("Akhir Tanggal", pd.to_datetime(conditions_df['Timestamp']).max())

    filtered_conditions = conditions_df[(pd.to_datetime(conditions_df['Timestamp']) >= pd.to_datetime(start_date)) &
                                        (pd.to_datetime(conditions_df['Timestamp']) <= pd.to_datetime(end_date))]
    st.write(filtered_conditions)

    # Line Chart for Sensor Data
    st.subheader("Grafik Data Sensor")
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(filtered_conditions['Timestamp']), filtered_conditions['Kelembaban (%)'], label='Kelembaban (%)')
    ax.plot(pd.to_datetime(filtered_conditions['Timestamp']), filtered_conditions['Suhu (°C)'], label='Suhu (°C)')
    fig.patch.set_alpha(0)  # Set figure background to transparent
    ax.set_facecolor((0, 0, 0, 0))  # Transparent background
    ax.set_title('Kelembaban dan Suhu dari Waktu ke Waktu', color='white')
    ax.set_xlabel('Waktu', color='white')
    ax.set_ylabel('Nilai', color='white')
    ax.tick_params(colors='white')
    ax.legend()
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # Set axis border to white
    st.pyplot(fig)

    # Bar Chart for Nutrient Levels
    st.subheader("Grafik Data Nutrisi Tanah")
    nutrient_df = filtered_conditions[["Nitrogen (ppm)", "Fosfor (ppm)", "Kalium (ppm)"]].mean()
    fig, ax = plt.subplots()
    nutrient_df.plot(kind='bar', ax=ax, color='blue')  # Set bar color to white
    fig.patch.set_alpha(0)  # Set figure background to transparent
    ax.set_facecolor((0, 0, 0, 0))  # Transparent background
    ax.set_title('Rata-rata Kandungan Nutrisi Tanah', color='white')
    ax.set_ylabel('ppm', color='white')
    ax.tick_params(colors='white')  # Set tick params to white
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # Set axis border to white
    st.pyplot(fig)


elif data_option == "Prakiraan Cuaca":
    st.header("Data Prakiraan Cuaca")

    # Filter options
    st.sidebar.subheader("Filter Data")
    selected_condition = st.sidebar.multiselect("Pilih Kondisi Cuaca:", forecast_df['Kondisi_cuaca'].unique(), default=forecast_df['Kondisi_cuaca'].unique())
    filtered_forecast = forecast_df[forecast_df['Kondisi_cuaca'].isin(selected_condition)]
    st.write(filtered_forecast)

    # Display mitigation recommendations
    st.subheader("Mitigasi Penyakit")
    mitigation = filtered_forecast["Mitigasi Penyakit"].value_counts()
    st.write(mitigation)

elif data_option == "Prediksi Penyakit":
    st.header("Data Prediksi Penyakit")

    # Filter options
    st.sidebar.subheader("Filter Data")
    selected_disease = st.sidebar.multiselect("Pilih Prediksi Penyakit:", predictions_df['Prediksi_Penyakit'].unique(), default=predictions_df['Prediksi_Penyakit'].unique())
    filtered_predictions = predictions_df[predictions_df['Prediksi_Penyakit'].isin(selected_disease)]
    st.write(filtered_predictions)

    # Pie Chart for Disease Prediction
    st.subheader("Distribusi Prediksi Penyakit")
    disease_counts = filtered_predictions["Prediksi_Penyakit"].value_counts()
    fig, ax = plt.subplots()
    disease_counts.plot.pie(autopct='%1.1f%%', ax=ax, textprops={'color': 'white'})  # Set text color to white
    fig.patch.set_alpha(0)  # Set figure background to transparent
    ax.set_facecolor((0, 0, 0, 0))  # Transparent background
    ax.set_title('Distribusi Prediksi Penyakit', color='white')
    ax.tick_params(colors='white')  # Set tick params to white
    ax.yaxis.label.set_color('white')  # Set y-axis label color
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # Set axis border to white
    st.pyplot(fig)


elif data_option == "Chatbot Pertanian":
    st.header("Chatbot Asisten Pertanian")
    st.write("Tanyakan apa saja tentang kondisi pertanian Anda!")

    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat input and response handling
    if prompt := st.chat_input("Masukkan pertanyaan Anda"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        chat_session.history.append({"role": "user", "parts": [prompt]})

        # Include latest data in the response context
        recent_conditions = conditions_df.iloc[-10:].to_dict(orient="records")
        recent_forecasts = forecast_df.iloc[-10:].to_dict(orient="records")
        recent_predictions = predictions_df.iloc[-10:].to_dict(orient="records")

        context = "Kondisi terbaru (10 data terakhir):\n"
        for i in range(10):
            context += (
                f"Data {i + 1}:\n"
                f"- Suhu: {recent_conditions[i]['Suhu (°C)']} °C\n"
                f"- Kelembaban: {recent_conditions[i]['Kelembaban (%)']} %\n"
                f"- pH Tanah: {recent_conditions[i]['pH Tanah']}\n"
                f"- Cuaca: {recent_forecasts[i]['Kondisi_cuaca']}\n"
                f"- Prediksi Penyakit: {recent_predictions[i]['Prediksi_Penyakit']}\n\n"
            )

        chat_session.history.append({"role": "model", "parts": [context]})
        pertanyaan = "Context :" + context +"\nPertanyaan : "+ prompt
        print(pertanyaan)
        # Get response from the model
        response = chat_session.send_message(pertanyaan)

        # Add model response to history
        st.session_state.messages.append({"role": "model", "parts": [response.text]})
        chat_session.history.append({"role": "model", "parts": [response.text]})

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message['parts'][0])
        else:
            st.chat_message("assistant", avatar="🌱").write(message['parts'][0])
