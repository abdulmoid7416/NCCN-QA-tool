import gradio as gr
import pinecone
from openai import OpenAI
import os
import json
import numpy as np
import csv
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-west1-gcp')

# Index name
index_name = 'clinical-diagram'

# Connect to the index
index = pinecone.Index(index_name)

def log_to_csv(data):
    filename = "clinical_qa_log.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Patient Description', 'QA Count', 'Question', 'Answer',
                             'Treatment Recommendations Count', 'NCCN Concordance', 
                             'Some Correct', 'Hallucinated Treatments'])
        
        writer.writerow(data)

# Global variables
patient_info = {"id": "", "description": ""}
qa_count = 0

# Define the embedding function
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Function to query Pinecone using combined embeddings
def query_pinecone(patient_description, user_question, top_k=3):
    # Get embeddings
    patient_embedding = get_embedding(patient_description)
    question_embedding = get_embedding(user_question)
    
    # Combine embeddings by averaging
    combined_embedding = np.mean([patient_embedding, question_embedding], axis=0)
    
    # Convert numpy array to list
    combined_embedding_list = combined_embedding.tolist()
    
    # Query Pinecone
    results = index.query(vector=combined_embedding_list, top_k=top_k, include_metadata=True)
    
    # Retrieve top k relevant diagrams
    relevant_diagrams = []
    for match in results['matches']:
        if 'metadata' in match and 'json' in match['metadata']:
            diagram_json = json.loads(match['metadata']['json'])
            # Make sure the image path is correctly extracted
            if 'image_path' in diagram_json:
                relevant_diagrams.append(diagram_json)
            else:
                print(f"Warning: No image path found in diagram: {diagram_json}")
        else:
            print(f"Warning: Expected metadata not found in match: {match}")
    
    return relevant_diagrams

# Function to answer user questions using relevant diagrams with GPT-4o
def answer_question(patient_description, user_question, diagrams):
    # Combine all relevant diagrams into a single context
    context = "\n".join([json.dumps(diagram) for diagram in diagrams])
    prompt = f"""
    You are an expert Oncologist assistant that answers questions based on the provided context.

    Patient Description: {patient_description}

    Question: {user_question}

    Context:
    {context}

    Please provide a detailed and accurate answer to the question based on the patient description and the context.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# Gradio Functions
def reset_patient():
    global patient_info, qa_count
    patient_info = {"id": "", "description": ""}
    qa_count = 0
    return (
        "",  # Clear patient description
        gr.update(value=""),  # Clear qa_desc
        gr.update(visible=True),  # Show patient input
        gr.update(visible=False),  # Hide qa interface
        gr.update(visible=False),  # Hide change patient button
        "",  # Clear question input
        "",  # Clear answer output
        None  # Clear image output
    )

def set_patient(patient_description):
    global patient_info, qa_count
    patient_info["description"] = patient_description
    qa_count = 0
    return (
        f"Patient Description: {patient_description}",
        gr.update(visible=True),
        gr.update(visible=False),
        "",  # Clear question input
        "",  # Clear answer output
        None,  # Clear image output
        gr.update(visible=False),  # Hide ask another question button
        gr.update(visible=False)   # Hide change patient button
    )

def qa_tool(user_question):
    global patient_info, qa_count
    relevant_diagrams = query_pinecone(patient_info["description"], user_question)
    if relevant_diagrams:
        answer = answer_question(patient_info["description"], user_question, relevant_diagrams)
        images = [diagram.get('image_path', 'No image path found') for diagram in relevant_diagrams[:3]]
        qa_count += 1

        # Log to CSV
        log_data = [
            patient_info["description"] if qa_count == 1 else "",
            qa_count,
            user_question,
            answer,
            "",  # Treatment Recommendations Count (placeholder)
            "",  # NCCN Concordance (placeholder)
            "",  # Some Correct (placeholder)
            ""   # Hallucinated Treatments (placeholder)
        ]
        log_to_csv(log_data)

        return answer, images, gr.update(visible=True), gr.update(visible=True)
    else:
        return "No relevant diagrams found with sufficient similarity.", [], gr.update(visible=True), gr.update(visible=True)

def ask_another_question():
    return "", "", "", gr.update(visible=True), gr.update(visible=False)

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# Clinical Diagram QA Tool")
    
    with gr.Group() as patient_input:
        patient_desc = gr.Textbox(label="Patient Description")
        set_patient_btn = gr.Button("Set Patient")

    with gr.Group() as qa_interface:
        qa_desc = gr.Markdown()
        question_input = gr.Textbox(label="Enter your question")
        get_answer_btn = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer")
        image_output = gr.Gallery(label="Relevant Images")
        ask_another_question_btn = gr.Button("Ask Another Question")
        change_patient_btn = gr.Button("Set Another Patient")

    qa_interface.visible = False
    ask_another_question_btn.visible = False
    change_patient_btn.visible = False

    set_patient_btn.click(
    set_patient,
    inputs=[patient_desc],
    outputs=[
        qa_desc,
        qa_interface,
        patient_input,
        question_input,
        answer_output,
        image_output,
        ask_another_question_btn,
        change_patient_btn
    ]
)

    get_answer_btn.click(
        qa_tool,
        inputs=[question_input],
        outputs=[answer_output, image_output, ask_another_question_btn, change_patient_btn]
    )

    ask_another_question_btn.click(
        ask_another_question,
        outputs=[question_input, answer_output, image_output, get_answer_btn, ask_another_question_btn]
    )

    change_patient_btn.click(
        reset_patient,
        outputs=[
            patient_desc,
            qa_desc,
            patient_input,
            qa_interface,
            change_patient_btn,
            question_input,
            answer_output,
            image_output
        ]
    )

if __name__ == "__main__":
    app.launch()
