# Import FastAPI framework to create the server and define API endpoints
from fastapi import FastAPI, Request, Form, HTTPException

# Import HTMLResponse to be able to return raw HTML pages
from fastapi.responses import HTMLResponse

# Import BaseModel from Pydantic to define request data models for validation
from pydantic import BaseModel

# Import HumanMessage class to wrap user input as a message for the agent
from langchain_core.messages import HumanMessage

# Import SQLChatMessageHistory to store conversation history in a local database (SQLite)
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Import uvicorn server to run FastAPI application locally
import uvicorn

# Import the graph object (likely your agent graph) from a local file called multi_agent_graph.py
from multi_agent_graph import graph

# Initialize a FastAPI app instance with a title
app = FastAPI(title="Medical Agent MCP Server")

# 📦 Define a request model for the JSON API endpoint
class QueryRequest(BaseModel):
    message: str                   # Field: the message that user wants to send
    session_id: str = "default"     # Optional Field: session_id, default value is "default"

# 🏠 Define the home route (/) that returns an HTML page
@app.get("/", response_class=HTMLResponse)
async def home():
    # Return a simple HTML page with a form, buttons, and dark mode toggle
    return """
    <html>
    <head>
        <title>Medical Agent MCP</title>
        <style>
            /* Styling the body: nice font, center content, background color */
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f2f2f2; color: #222; margin: 0; padding: 0; height: 100vh; display: flex; justify-content: center; align-items: center; transition: 0.3s; }
            /* Dark mode styles: change background and text color */
            .dark-mode { background: #121212; color: #f1f1f1; }
            /* Styling the container: white card with shadow and padding */
            .container { background: #fff; padding: 30px 40px; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); width: 100%; max-width: 500px; text-align: center; transition: 0.3s; }
            /* Dark mode container background */
            .dark-mode .container { background: #1f1f1f; }
            /* Styling the inputs and buttons */
            input[type="text"], button, input[type="submit"] { width: 100%; padding: 12px; margin-top: 10px; border-radius: 5px; font-size: 16px; }
            input[type="text"] { border: 1px solid #ccc; }
            input[type="submit"] { background: #0d6efd; color: white; border: none; }
            input[type="submit"]:hover { background: #0b5ed7; }
            /* Styling clear and dark mode buttons */
            .clear-btn { background: #6c757d; color: white; border: none; }
            .clear-btn:hover { background: #5a6268; }
            .dark-toggle-btn { background: #198754; color: white; border: none; }
            .dark-toggle-btn:hover { background: #157347; }
        </style>
        <script>
            // Toggle dark mode by adding/removing a class and saving the preference
            function toggleDarkMode() {
                document.body.classList.toggle("dark-mode");
                localStorage.setItem("darkMode", document.body.classList.contains("dark-mode"));
            }
            // Clear the input field
            function clearInput() {
                document.getElementById("messageInput").value = "";
            }
            // When page loads, check if dark mode is saved, and apply it
            window.onload = () => {
                if (localStorage.getItem("darkMode") === "true") {
                    document.body.classList.add("dark-mode");
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Welcome to Medical Agent MCP</h1>
            <form action="/process_form" method="post">
                <!-- Input field for user to write the message -->
                <input id="messageInput" type="text" name="message" placeholder="Describe your symptoms..." required>
                <!-- Submit button to send the form -->
                <input type="submit" value="Ask Agent">
                <!-- Clear input button -->
                <button type="button" class="clear-btn" onclick="clearInput()">🧹 Clear</button>
                <!-- Dark mode toggle button -->
                <button type="button" class="dark-toggle-btn" onclick="toggleDarkMode()">🌗 Dark Mode</button>
            </form>
        </div>
    </body>
    </html>
    """

# 📩 Handle the form submission and display the agent's response
@app.post("/process_form", response_class=HTMLResponse)
async def handle_form(message: str = Form(...)):
    try:
        # Create a memory object to track the chat history, using SQLite database
        memory = SQLChatMessageHistory(session_id="form_session", connection="sqlite:///chat_history.db")
        
        # Invoke the graph (agent) passing the message as a HumanMessage, memory object, and an empty response
        result = graph.invoke({"messages": [HumanMessage(content=message)], "memory": memory, "response": ""})
        
        # Return a new HTML page showing the original message and agent's response
        return f"""
        <html>
        <head>
            <title>Agent Response</title>
            <style>
                /* Styling similar to homepage but for result page */
                body {{ font-family: Arial, sans-serif; background: #f0f0f0; color: #000; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; transition: 0.3s; }}
                .dark-mode {{ background: #121212; color: #eee; }}
                .card {{ background: white; padding: 20px 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .dark-mode .card {{ background: #1e1e1e; }}
                button {{ margin-top: 20px; padding: 10px 20px; background: #0d6efd; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }}
                button:hover {{ background: #0b5ed7; }}
            </style>
            <script>
                // Same dark mode toggle logic
                function toggleDarkMode() {{
                    document.body.classList.toggle("dark-mode");
                    localStorage.setItem("darkMode", document.body.classList.contains("dark-mode"));
                }}
                window.onload = () => {{
                    if (localStorage.getItem("darkMode") === "true") {{
                        document.body.classList.add("dark-mode");
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="card">
                <h2>🧠 Agent Response</h2>
                <p><strong>Message:</strong> {message}</p>
                <p><strong>Response:</strong> {result['response']}</p>
            </div>
            <!-- Button to go back home -->
            <button onclick="window.location.href='/'">⬅️ Back to Home</button>
            <!-- Button to toggle dark mode -->
            <button onclick="toggleDarkMode()">🌗 Toggle Dark Mode</button>
        </body>
        </html>
        """
    except Exception as e:
        # In case of any error, return an error message with a 500 status code
        return HTMLResponse(f"<p>Error: {str(e)}</p><a href='/'>⬅️ Back</a>", status_code=500)

# 🔥 Define an endpoint to process JSON POST requests (API calls)
@app.post("/process")
async def handle_json(request: QueryRequest):
    try:
        # Create a memory object for the session based on session_id provided in request
        memory = SQLChatMessageHistory(session_id=request.session_id, connection="sqlite:///chat_history.db")
        
        # Invoke the agent (graph) with the message
        result = graph.invoke({"messages": [HumanMessage(content=request.message)], "memory": memory, "response": ""})
        
        # Return a JSON response containing the agent's output, session_id, and a success status
        return {"response": result["response"], "session_id": request.session_id, "status": "success"}
    except Exception as e:
        # If there's an error, raise an HTTPException with 500 Internal Server Error
        raise HTTPException(status_code=500, detail=str(e))

# ❤️ Define a simple health check route to see if the server is alive
@app.get("/health")
async def health():
    # Return a simple JSON saying the app is healthy
    return {"status": "healthy"}

# 🚀 Run the server locally when the script is executed directly
if __name__ == "__main__":
    # Start uvicorn server, hosting on 0.0.0.0 to be accessible from any device on the network, reload=True for development
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
