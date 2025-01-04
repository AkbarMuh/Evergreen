import os  
import base64
from openai import AzureOpenAI  

endpoint = os.getenv("ENDPOINT_URL", "https://openai-edasquad2.openai.azure.com/")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "d007d2dca97a4c7bbce8792a431e25cc")  

# Initialize Azure OpenAI client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",  
)
    

#Prepare the chat prompt 
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "buatkan gambar gunung"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Saya tidak dapat membuat gambar secara langsung. Namun, saya bisa membantu Anda menggambarkan gunung atau memberikan ide tentang cara membuat gambar gunung. \n\nJika Anda ingin menggambar gunung, berikut adalah langkah-langkahnya:\n\n1. *Bentuk Dasar: Mulailah dengan menggambar bentuk segitiga besar untuk puncak gunung. Anda bisa membuat beberapa puncak untuk tampilan yang lebih dramatis.\n\n2. **Detail Puncak: Tambahkan garis-garis zig-zag di bagian atas untuk menunjukkan salju atau bebatuan.\n\n3. **Lereng Gunung: Gambar garis-garis diagonal dari puncak ke dasar untuk menunjukkan lereng. Anda bisa membuatnya lebih curam atau landai sesuai keinginan.\n\n4. **Vegetasi: Tambahkan beberapa pohon atau semak-semak di lereng gunung untuk memberikan kesan alami.\n\n5. **Latar Belakang: Gambar langit di belakang gunung. Anda bisa menambahkan matahari terbenam atau awan untuk efek yang lebih menarik.\n\n6. **Warna*: Warnai gambar menggunakan pensil warna atau cat. Gunung bisa berwarna abu-abu, coklat, dan putih di puncaknya, sedangkan latar belakang bisa berwarna biru untuk langit.\n\nJika Anda ingin gambar gunung yang sudah jadi, Anda bisa mencari di situs web penyedia gambar gratis seperti Unsplash atau Pixabay."
            }
        ]
    }
] 
    
# Include speech result if speech is enabled  
messages = chat_prompt  
    
# Generate the completion  
completion = client.chat.completions.create(  
    model=deployment,  
    messages=messages,  
    max_tokens=800,  
    temperature=0.7,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,  
    stop=None,  
    stream=False
)

print(completion.to_json())