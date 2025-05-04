<h1 align="center">⚙️ TensoScan Backend</h1>

> **Backend services powering the AI-driven blood pressure monitoring ecosystem**
>
> This backend provides the essential APIs and infrastructure to support the TensoScan mobile application, focusing on secure data management, user authentication, and enabling intelligent features for blood pressure tracking

---

## 🚀 Key Features

- 🔑 **Secure User Authentication and Authorization**: Protecting user data with robust authentication mechanisms
- 💾 **Centralized Data Storage**: Utilizing a scalable database for storing and managing blood pressure readings and user information
- 🔗 **Comprehensive API Endpoints**: Offering well-defined and secure APIs for seamless communication with the TensoScan mobile application
- 🧠 **Integration with AI/ML Capabilities**: Laying the groundwork for future intelligent features, potentially leveraging models for data analysis or insights
- ⚙️ **Scalable and Robust Architecture**: Designed to handle a growing user base and increasing data loads

---

## 🧱 Architecture

- **API Layer (FastAPI/Starlette)**: Handles incoming requests from the mobile application
- **Authentication Layer (python-jose, Passlib)**: Manages user registration, login, and token-based authentication
- **Business Logic Layer**: Contains the core application logic for data processing and management
- **Data Access Layer (SQLAlchemy/MongoDB)**: Interacts with the database for data persistence
- **Potential ML Integration**: Modules for interacting with machine learning models (TensorFlow, Keras, OpenCV)

---

## 🧰 Tech Stack

| Category         | Technology                                  | Notes                                                                 |
|------------------|---------------------------------------------|-----------------------------------------------------------------------|
| ⚙️ Backend       | FastAPI (~0.114.2), Starlette (0.38.6), Uvicorn (0.34.1) | High-performance asynchronous framework and server                 |
| 🐍 Language       | Python                                      |                                                                       |
| 🔑 Authentication | python-jose, Passlib (~1.7.4), bcrypt (~4.3.0) | JWT for token management and password hashing                        |
| 💾 Database      | SQLAlchemy (likely), pymongo (3.12.0)       | Potential for relational (SQLAlchemy) or NoSQL (MongoDB) database |
| 🌐 Networking    | httpx (~0.28.1), requests (2.32.3)          | For making HTTP requests                                             |
| 🛡️ Security       | cryptography (44.0.2)                         | Provides cryptographic primitives                                      |
| ⚙️ Utilities      | python-dotenv (1.0.1), pydantic (2.11.3)      | Configuration management and data validation                         |
| 🧠 ML/AI          | tensorflow (2.19.0), keras (3.9.2), opencv-python (~4.11.0.86), numpy (1.26.4), scikit-image (~0.24.0) | Libraries for potential AI/ML features                             |
| 📊 Data & Science | scipy (1.13.1), matplotlib (3.9.4), plotly (6.0.1) | For data manipulation, scientific computing, and visualization      |
| 📄 PDF Handling   | fpdf (1.7.2), pikepdf (9.7.0), pypdf (5.4.0), reportlab (4.4.0), xhtml2pdf (0.2.17) | Libraries for generating and manipulating PDF documents              |
| 🖼️ Image Processing| Pillow (~10.0.1), imageio (2.37.0)         | For image manipulation                                               |

---

## 🔗 API Endpoints

🔐 Authentication
| Method | Endpoint    | Description                            | Auth Required |
| ------ | ----------- | -------------------------------------- | ------------- |
| POST   | `/register` | Register a new user                    | ❌ No          |
| POST   | `/login`    | Authenticate user and return JWT token | ❌ No          |


💬 Chatbot Interaction
| Method | Endpoint   | Description                                                        | Auth Required |
| ------ | ---------- | ------------------------------------------------------------------ | ------------- |
| POST   | `/chatbot` | Get AI-generated response based on prompt and conversation history | ✅ Yes         |


🖼️ Image Recognition
| Method | Endpoint             | Description                                                            | Auth Required |
| ------ | -------------------- | ---------------------------------------------------------------------- | ------------- |
| POST   | `/recognize`         | Recognize numbers from uploaded image (raw model)                      | ❌ No          |
| POST   | `/display-recognize` | Analyze image, return structured blood pressure result, and save to DB | ✅ Yes         |


📊 Measurements Management
| Method | Endpoint                    | Description                               | Auth Required |
| ------ | --------------------------- | ----------------------------------------- | ------------- |
| GET    | `/measurements`             | Get user's historical blood pressure data | ✅ Yes         |
| DELETE | `/remove/measurements/{id}` | Delete a specific measurement by ID       | ✅ Yes         |


🧾 Reports & Visualization
| Method | Endpoint                  | Description                                     | Auth Required |
| ------ | ------------------------- | ----------------------------------------------- | ------------- |
| GET    | `/user/measurements/html` | Get user's measurements rendered in HTML format | ✅ Yes         |
| GET    | `/user/measurements/pdf`  | Download user's measurements as a PDF document  | ✅ Yes         |

---

## 🧠 Potential AI/ML Integration

- **Data Analysis**: Analyzing blood pressure trends and patterns
- **Insights Generation**: Providing users with personalized insights based on their historical data
- **Future Features**: Exploring more advanced AI-powered functionalities

---

## 📄 License

This project is licensed under the MIT License
