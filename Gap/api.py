
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import create_rag, query_rag
from graph import app_graph
# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/api/initialize', methods=['GET','POST'])
def initialize_system():
    """Initialize the RAG system"""  
    try:

        create_rag()
        print("success initializing system")
        return jsonify({
            "success": True,
             "status": "successfully initialized system"
        })

    except Exception as e:
        print(f"Error initializing system: {e}")
        return jsonify({
            "success": False,
            "status": "error initializing system",
            "error": str(e)
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        # data = request.get_json()
        # question = data.get('question', '').strip()
        # question = request.args.get('question', '').strip()
        # print("===========start====================")
        data_req = request.get_json()

        if not data_req:
            return jsonify({"error": "No request data received"}), 400

        taktzivim = data_req.get('taktzivimLemosad', [])
        maanimWithKriteryonim = data_req.get('maanimWithKriteryonim', [])
        question = data_req.get('question', '').strip()
        chat_history = data_req.get('chat_history', [])
        # print("================finish get data========================")
        # תוכל לשמור את זה, לעבד או לאתחל את המערכת איתם
        # לדוגמה:
        # print("Taktzivim:", taktzivim)
        # print("Maanim:", maanimWithKriteryonim)

        
        if not question:
            return jsonify({
                "error": "Question not found",
                "answer": "Please enter a valid question"
            }), 400
        user_info = taktzivim
        # Create initial state
        initial_state = {
            "messages": [],
            "question": question,
            # "vectorstore": current_vectorstore,
            "retrieved_docs": [],
            "answer": "",
            "search_query": "",
            "sources": [],
            "user_info": user_info,
            "chat_history": chat_history,
            "summary": ""
        }
        print(f"----------Processing question: {question}-----------")
        # Run the workflow
        result = app_graph.invoke(initial_state)
        
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "question": question,
            "search_query": result["search_query"],
            "summary": result["summary"],
            "chat_history": result["chat_history"],
        })
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({
            "error": "Error processing question",
            "answer": "Sorry, an error occurred while processing the question. Please try again."
        }), 500

if __name__ == '__main__':
    print("Starting RAG Backend...")
    app.run(debug=True, host='0.0.0.0', port=5000)