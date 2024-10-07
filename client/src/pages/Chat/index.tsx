import React, { useState, useRef, useEffect } from "react";
import { Input, Button, Avatar, Row, Col, message as AntMessage } from "antd";
import "./ChatPage.css"; 
import axios from 'axios';


type Message = {
  content: string;
  sender: "user" | "bot"; 
};

const ChatPage: React.FC = () => {
  const [message, setMessage] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null); 
  const [loading, setLoading] = useState<boolean>(false); 


  const sendmsg = async (msg: string) => {
    try {
      setLoading(true);
      const apiUrl = import.meta.env.VITE_BACKEND_URL;

      const response = await axios.post(`${import.meta.env.VITE_BACKEND_URL}/chat`, {
        query: msg
      });

      if (response.data.message) {
        setChatHistory(prev => [
          ...prev,
          { content: response.data.message, sender: "bot" }
        ]);
      } else {
        AntMessage.error("No response from the bot.");
      }
    } catch (error) {
      console.error("Error sending message:", error);
      AntMessage.error("Failed to communicate with the server.");
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (message.trim() === "") {
      return; 
    }

    setChatHistory(prev => [...prev, { content: message, sender: "user" }]);
    

    setMessage("");
    await sendmsg(message);
  };


  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  return (
    <div className="chat-container">
      <div className="chat-history">
        {chatHistory.map((msg, index) => (
          <Row
            key={index}
            justify={msg.sender === "user" ? "start" : "end"}
            className="chat-message-row"
          >
            {msg.sender === "user" && (
              <Col>
                <Avatar style={{ backgroundColor: "#87d068" }} icon="U" />
              </Col>
            )}
            <Col>
              <div
                className={`chat-bubble ${
                  msg.sender === "user" ? "chat-bubble-user" : "chat-bubble-bot"
                }`}
              >
                {msg.content}
              </div>
            </Col>
            {msg.sender === "bot" && (
              <Col>
                <Avatar style={{ backgroundColor: "#f56a00" }} icon="B" />
              </Col>
            )}
          </Row>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-container">
        <div className="chat-input-section">
          <Input
            placeholder="Type your message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onPressEnter={sendMessage}
            style={{ flex: 1 }}
          />
          <Button type="primary" onClick={sendMessage}>
            Send
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
