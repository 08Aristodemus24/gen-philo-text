import { Fragment, useState } from "react";
import Message from "./Message";


function getCookie(name){
    let cookieValue = null;
    if (document.cookie && document.cookie !== ''){
        const cookies = document.cookie.split(';');
        for(let i = 0; i < cookies.length; i++){
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if(cookie.substring(0, name.length + 1) === (name + '=')){
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

export default function Glass(){
    // states
    let [messages, setMessages] = useState([]);
    let [prompt, setPrompt] = useState('');
    let [temperature, setTemperature] = useState(1.0);
    let [seqLen, setSeqLen] = useState(250);
    console.log(seqLen);

    // build list of message components
    const Messages = messages.map((value, index) => {
        return <Message key={index} message={value}/>
    });

    // send post request with prompt and temperature
    // to make predictions to model
    const csrf_token = getCookie('csrftoken');
    const generate = async (event) => {
        try{
            event.preventDefault();

            // const url = "https://gen-philo-text.onrender.com/predict";
            const url = "http://127.0.0.1:5000/predict";
            const response = await fetch(url, {
                method: 'POST',
                body: JSON.stringify({
                    prompt: prompt,
                    temperature: temperature,
                    sequence_length: seqLen
                }),
                headers: { 
                    'X-CSRFToken': csrf_token,
                    'Content-Type': 'application/json; charset=UTF-8'
                }
            });

            // extract the message from .json()
            const data = await response.json();

            // if response.status is 200 then that means contact information
            // has been successfully sent to the email.js api
            if(response.status === 200){
                console.log(`message has been sent with code ${response.status}`);

                // once message has been received set the state 
                // such that new message is appended to previous
                // set of messages
                console.log(data);
                setMessages((messages) => {
                    return [...messages, data];
                });                

            }else{
                console.log(`message submission unsucessful. Response status '${response.status}' occured`);

            }

        }catch(error){
            console.log(`Submission denied. Error '${error}' occured`);
        }
    };

    return (
        <div className="glass-container">
            <div className="glass">
                <div className="messages-container">
                    {Messages}
                </div>
                <div className="prompts-container">
                    <div className="prompt-group">
                        <div className="input-container">
                            <label htmlFor="prompt-field" className="prompt-label">prompt: </label>
                            <input onChange={(event) => setPrompt(event.target.value)} value={prompt} id="prompt-field" className="prompt-field" type="text" placeholder="Type a prompt e.g. Dostoevsky"/>
                        </div>
                        
                        <div className="input-container">
                            <label htmlFor="temp-field" className="temp-label">temperature: </label>
                            <input onChange={(event) => setTemperature(event.target.value)} type="range" value={temperature} min={0.0} max={2.0} step={0.01} id="temp-field"  className="temp-field"/>
                        </div>
                        
                        <div className="input-container">
                            <label htmlFor="seq-len-field" className="seq-len-label">sequence length: </label>
                            <input onChange={(event) => setSeqLen(event.target.value)} type="number" id="seq-len-field" className="seq-len-field" placeholder={seqLen}/>
                        </div>
                    </div>

                    <button onClick={generate} className="generate-btn">generate</button>
                </div>
            </div>
        </div>
    );
}