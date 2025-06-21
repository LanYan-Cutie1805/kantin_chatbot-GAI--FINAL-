# 🍱 Kantin P Chatbot – Search Food by Image

This is the **second chatbot** in the Kantin P Chatbot series, designed to help users **search for similar foods** in **Kantin Gedung P, Universitas Kristen Petra**, by uploading a picture of food.

## 📌 Features

- 📷 **Search Menus by Food Photo**
  - Upload any food image (e.g., soup, fried rice, meat, drinks)
  - The chatbot will:
    - 🧠 Analyze the food using **Llama 3.2 Vision**
    - ✍️ Generate a food description
    - 🔎 Search the Kantin menu for **similar dishes**
    - 🍛 Return matching dish names from the menu

- 🧠 **Vision-Language Model**
  - Powered by **Llama 3.2 Vision** via [Ollama](https://ollama.com)
  - Uses **FastEmbed** for embeddings
  - Menu data is loaded from a **CSV file**, not a database
