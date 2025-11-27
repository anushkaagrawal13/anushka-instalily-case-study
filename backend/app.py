# app.py
# backend/app.py
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from vector_manager import VectorManager
import os
import re
import json

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SYSTEM_PROMPT = """
You are a PartSelect Assistant. ONLY answer questions about refrigerator or dishwasher parts,
installation steps, troubleshooting, compatibility, and ordering support.
If a question is out of scope, respond: "I'm only able to help with refrigerator and dishwasher parts."
Include product_card JSON when relevant: {part_number, name, brand, image, link}.
"""

app = Flask(__name__)
vm = VectorManager()
vm.index_documents()
retriever = vm.get_retriever()

# Minimal intent classifier
def detect_intent(user_text):
    # naive simple rules
    user_text = user_text.lower()
    if any(w in user_text for w in ["install", "installation"]):
        return "installation"
    if any(w in user_text for w in ["compatible", "model"]):
        return "compatibility"
    if any(w in user_text for w in ["not working", "broken", "error", "issue"]):
        return "troubleshooting"
    return "part_lookup"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")
    intent = detect_intent(user_msg)

    prompt = PromptTemplate(
        input_variables=["system_prompt", "context", "question"],
        template="{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer in markdown. Include product_card JSON if relevant."
    )

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    answer = qa.run(user_msg)

    # extract JSON product_card if present
    product_card = None
    m = re.search(r"```json\s*({.*?})\s*```", answer, re.S)
    if m:
        try:
            product_card = json.loads(m.group(1))
        except:
            product_card = None

    return jsonify({"answer": answer, "intent": intent, "product_card": product_card})

if __name__ == "__main__":
    app.run(port=5000, debug=True)