import streamlit as st
import emoji
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.readers.file import PyMuPDFReader
from qdrant_client.http.models import VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
import pandas as pd
import re
from PIL import Image
import ast
import glob

import nest_asyncio
nest_asyncio.apply()


def chatbot_kantin():
    # Initialize Qdrant Client
    from qdrant_client import QdrantClient
    QDRANT_URL = "INSERT_HERE"
    QDRANT_API_KEY = "INSERT_HERE"
    
    
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    COLLECTION_NAME = "CSV_Data"
    collections = qdrant_client.get_collections()
    print(collections)
    
    # initialize node parser
    splitter = SentenceSplitter(chunk_size=512)
    
    import sys
    import logging
    
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    
    system_prompt = """
    You are a multi-lingual expert system who has knowledge, based on 
    real-time data. You will always try to be helpful and try to help them 
    answering their question. If you don't know the answer, say that you DON'T
    KNOW.
    
    Jawablah semua dalam Bahasa Indonesia.
    Tugas Anda adalah untuk menjadi pelayan kantin yang ramah yang dapat mengarahkan user.
    Anda tidak melayani pemesanan.
    
    Kantin yang Anda layani adalah kantin kampus Universitas Kristen Petra Surabaya.
    Pada Universitas Kristen Petra terdapat 2 gedung utama yang setiap gedungnya memiliki kantin, 
    yaitu Gedung P dan W.
    
    Arahkanlah mahasiswa dan staff yang lapar ke kantin dan ke stall kantin yang tepat
    berdasarkan keinginan mereka. Berikanlah beberapa makanan dan minuman
    yang relevan berdasarkan kebutuhan mereka.
    
    Perhatikan perbedaan antara beberapa makanan, sebagai contoh, nasi ayam goreng memiliki implikasi menggunakan nasi putih sebagai dasar, sementara nasi goreng ayam memiliki dasar nasi goreng dengan lauk ayam.
    Hanya jawab dengan makanan/minuman yang relevan sesuai yang diminta.
    
    Untuk setiap jawaban, pastikan Anda memberikan detil yang lengkap.
    
    Percakapan sejauh ini:
    """
    
    Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
    Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")
    
    
    @st.cache_resource(show_spinner="Mempersiapkan data kantin – sabar ya.")
    def load_data(vector_store=None):
        with st.spinner(text="Mempersiapkan data kantin – sabar ya."):
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            index_retriever = index.as_retriever(similarity_top_k=8)
        
        return index_retriever
    
    # Function to get image path from the CSV file
    df = pd.read_csv("./docs/menu-kantin-2.csv")  # Load CSV globally
    def show_character_image(food_name):
        """Retrieve and display all images that match a given food keyword from the CSV."""
        matching_rows = df[df["Nama Produk"].str.lower().str.contains(food_name.lower(), na=False)]
    
        if not matching_rows.empty:
            image_paths = []
            for _, row in matching_rows.iterrows():
                image_path = row["Gambar"].strip()  # Get image path from CSV
                abs_path = os.path.abspath(image_path)  # Ensure correct path
    
                if os.path.exists(abs_path):
                    image_paths.append(abs_path)  # Store valid paths
                else:
                    st.warning(f"⚠️ Image not found: {abs_path}")
    
            if image_paths:
                return image_paths
            else:
                return None
        else:
            st.error(f"⚠️ No data found for {character_name}")
            return None
        
    
    def search_food_image(food_name):
        """
        Searches for all images related to a given food item in the local folder.
        """
        image_folder = "./images"  # Change this to your actual image folder
        search_pattern = os.path.join(image_folder, f"*{food_name}*.jpg")  # Adjust extension if needed
        
        image_paths = glob.glob(search_pattern)  # Find all matching images
        
        return image_paths if image_paths else []  # Always return a list
    
    def extract_food_names(response_text):
        """
        Extracts food names dynamically from the chatbot's response.
        """
        food_names = []
        known_foods = df["Nama Produk"].str.lower().unique()  # Get food names from CSV
        
        for food in known_foods:
            response_text = str(response_stream)
            if food in response_text.lower():
                food_names.append(food)
    
        return food_names
    
    # Main Program
    st.title("Petranesian Lapar 🍕:tropical_drink::coffee: :rice: :poultry_leg:")
    st.write("Chatbot untuk menu makanan di kantin Universitas Kristen Petra Gedung P dan W.")
    retriever = load_data()
    
    # Initialize chat history if empty
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Halo! Mau makan atau minum apa? 😉"}
        ]
    
    # Initialize the chat engine
    if "chat_engine" not in st.session_state.keys():
        # Initialize with custom chat history
        init_history = [
            ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Mau makan atau minum apa? 😉"),
        ]
        memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
        st.session_state.chat_engine = CondensePlusContextChatEngine(
            verbose=True,
            system_prompt=system_prompt,
            context_prompt=(
                    "Anda adalah pelayan kantin profesional yang ramah yang dapat mengarahkan user ketika mencari makanan dan stall kantin.\n"
                    "Format dokumen pendukung: gedung letak kantin, nama stall, nama produk, harga, keterangan\n"
                    "Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n"
                    "{context_str}"
                    "\n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna. Hanya jawab dengan kantin/menu yang sesuai. Jika tidak menemukan makanan atau minuman yang sesuai, maka katakan bahwa tidak menemukan."
                ),
            condense_prompt="""
    Diberikan suatu percakapan (antara User dan Assistant) dan pesan lanjutan dari User,
    Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
    dari percakapan sebelumnya. Pertanyaan independen/standalone question cukup 1 kalimat saja. Informasi yang penting adalah makanan/minuman yang dicari, nama stall, dan letak gedung. Contoh standalone question: "Saya mencari jus jambu di Gedung P".
    
    <Chat History>
    {chat_history}
    
    <Follow Up Message>
    {question}
    
    <Standalone question>""",
            memory=memory,
            retriever=retriever,
            llm=Settings.llm
        )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_image_path" not in st.session_state:
        st.session_state.last_image_path = None
    if "last_character" not in st.session_state:
        st.session_state.last_character = None
    if "displayed_images" not in st.session_state:
        st.session_state.displayed_images = set()  # Store displayed images
    
    def load_food_data():
        df = pd.read_csv("./docs/menu-kantin-2.csv")  # Ensure the CSV contains columns: Nama Produk, Harga, Nama Stall, Gambar
        food_dict = {row["Nama Produk"].lower(): (row["Harga"], row["Nama Stall"], row["Gambar"]) for _, row in df.iterrows()}
        return food_dict
    
    food_data = load_food_data()
    
    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                image_paths = eval(msg["content"])  # Convert stored string back to list
                captions = [f"{msg['food_name']} | {msg.get('stall_name', 'Unknown Stall')} | Rp {msg.get('price', 'N/A')}" for _ in image_paths]
                st.image(image_paths, caption=captions, use_column_width=True)
            else:
                st.markdown(msg["content"])
    
    # ✅ Now handle new user input
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
    
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        trigger_words = ["tunjukkan", "tunjukan", "hasilkan", "berikan", "mana"]
        image_words = ["gambar", "foto"]
    
        # ✅ Check if prompt requires an image
        if any(word in prompt.lower() for word in trigger_words) and any(img_word in prompt.lower() for img_word in image_words):
            cleaned_prompt = prompt.lower()
            for word in trigger_words + image_words:
                cleaned_prompt = cleaned_prompt.replace(word, "")
    
            cleaned_prompt = re.sub(r"\b(dari|nya|anu)\b", "", cleaned_prompt).strip()
            food_name = cleaned_prompt.title()
    
            if food_name.lower() in food_data:
                price, stall_name, image_path = food_data[food_name.lower()]
                
                # ✅ Store image details including price and stall name
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "image",
                    "content": str([image_path]),  # Store as a list for compatibility
                    "food_name": food_name,
                    "price": price,
                    "stall_name": stall_name
                })
    
                with st.chat_message("assistant"):
                    caption = f"{food_name} | Stall: {stall_name} | Rp {price}"
                    st.image(image_path, caption=caption, use_column_width=True)
            else:
                st.error("❌ Makanan tidak ditemukan dalam database.")
    
        else:
            # ✅ Handle text-based responses
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Loading..."):
                    placeholder.image("paimon-think.jpg", width=200)
                    response_stream = st.session_state.chat_engine.stream_chat(prompt)
                    st.write_stream(response_stream.response_gen)
                    response = st.session_state.chat_engine.chat(prompt)
                    placeholder.markdown(response_stream)
    
            # ✅ Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": response_stream})
    
            # ✅ Check for food names in the response
            food_names = extract_food_names(response_stream)
            for food_name in food_names:
                matched_food = None
                for stored_food in food_data.keys():
                    if food_name.lower() in stored_food:
                        matched_food = stored_food
                        break
                if matched_food:
                    price, stall_name, image_path = food_data[food_name.lower()]
    
                    # DEBUG
                    print(f"Debug Info - Food Name: {food_name}")
                    print(f"Debug Info - Image Path: {image_path}")
                    print(f"Debug Info - Stall Name: {stall_name}")
                    print(f"Debug Info - Price: {price}")
                    
                    with st.chat_message("assistant"):
                        caption = f"{food_name} | Stall: {stall_name} | Rp {price}"
                        st.image(image_path, caption=caption, use_column_width=True)
    
                    # ✅ Store image details
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "image",
                        "content": str([image_path]),
                        "food_name": food_name,
                        "price": price,
                        "stall_name": stall_name
                    })
    
            placeholder.empty()


def image_annotation_interface():
    from PIL import Image
    from io import BytesIO
    import pandas as pd
    import ollama
    from ollama import generate
    import os
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.core.schema import Document

    Settings.embed_model = FastEmbedEmbedding("intfloat/multilingual-e5-large", cache_dir="./fastembed_cache")

    def load_csv_documents(csv_path="docs/menu-kantin-2.csv"):
        df = pd.read_csv(csv_path)
        docs = []
    
        for _, row in df.iterrows():
            nama_produk = row.get("Nama Produk", "")
            nama_stall = row.get("Nama Stall", "")
            kantin = row.get("Kantin", "")
    
            full_text = f"{nama_produk} - {nama_stall} - {kantin}"
            metadata = {
                "Nama Produk": nama_produk,
                "Nama Stall": nama_stall,
                "Kantin": kantin
            }
    
            docs.append(Document(text=full_text, metadata=metadata))
    
        return docs
    
    def create_index_from_csv(csv_path="docs/menu-kantin-2.csv"):
        docs = load_csv_documents(csv_path)
        index = VectorStoreIndex.from_documents(docs)
        return index

    
    # 1. Set up LLM + Embedding model
    Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434")
    Settings.embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large", cache_dir="./fastembed_cache")
    

    
    def query_kantin_with_text(query_text, index):
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query_text)
    
        results = []
        for node in nodes:
            meta = node.metadata
            nama = meta.get("Nama Produk", "Unknown Produk")
            stall = meta.get("Nama Stall", "Unknown Stall")
            kantin = meta.get("Kantin", "Unknown Kantin")
    
            results.append(f"🍛 {nama} — Stall: {stall} | Kantin: {kantin}")
    
        return results
    index = create_index_from_csv()

    # Path to CSV
    CSV_PATH = 'image_descriptions.csv'
    
    # Load or create a DataFrame
    def load_or_create_dataframe(filename):
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=['image_file', 'description'])
        return df
    
    # Image processing function using Ollama
    def describe_uploaded_image(uploaded_image):
        image = Image.open(uploaded_image)
        with BytesIO() as buffer:
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            prompt = """
Lihat gambar ini dan tebak **nama makanan khas Indonesia** yang paling mendekati. 
Ikuti format ini **tanpa tambahan penjelasan apapun**:
\n
**\nNama Makanan/minuman:** (isikan di sini)

**Ciri-ciri:** (jelaskan warna, bahan utama, cara penyajian, atau detail lainnya)

Contoh:
**Nama Makanan/minuman:** Soto Ayam  
**Ciri-ciri:** Kuah kuning, suwiran ayam, telur rebus, sambal, seledri, dan bawang goreng.

Sekarang, jawablah berdasarkan gambar ini:
"""
        full_response = ''
        for response in generate(
            model='llama3.2-vision:11b-instruct-q4_K_M',
            prompt=prompt,
            images=[image_bytes],
            stream=True
        ):
            full_response += response['response']
    
        return full_response
    
    st.title("Cari Makanan dengan Gambar :coffee:")

    uploaded_file = st.file_uploader("Upload gambar makanan", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="", use_column_width=True)
        placeholder = st.empty()
        with st.spinner("Loading..."):
            
            placeholder.image("escoffier_think.png", width=200)
            description = describe_uploaded_image(uploaded_file)
            placeholder.empty()

            
            st.markdown(f" {description}")

        
        # Send the description to the chatbot
        similar_food_response = query_kantin_with_text(description, index)
        
        # Show the result from chatbot
        st.markdown("**Makanan yang mirip:**")
        for i, name in enumerate(similar_food_response):
            st.markdown(f"- {i+1}. **{name}**")



        # Optional: Return the description to use in other parts
        return description
        




###########################################################################################################################################
# Sidebar dropdown menu (cleaner, no radio dots)
st.sidebar.title("Petranesian Lapar")
selected_chatbot = st.sidebar.selectbox("Pilih chatbot:", ["Chatbot Kantin", "Cari Makanan dengan Gambar"])

if selected_chatbot == "Chatbot Kantin":
    st.subheader("")
    chatbot_kantin()
    # Your existing Kantin chatbot logic goes here

elif selected_chatbot == "Cari Makanan dengan Gambar":
    st.subheader("")
    image_annotation_interface()
    # Your second chatbot logic here
