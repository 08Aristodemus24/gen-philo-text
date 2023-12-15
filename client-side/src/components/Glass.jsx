import { Fragment, useState } from "react";

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
    console.log(prompt);

    // send post request with prompt and temperature
    // to make predictions to model
    const csrf_token = getCookie('csrftoken');
    const generate = async (event) => {
        try{
            event.preventDefault();

            const url = 'http://127.0.0.1:5000/predict';
            const response = await fetch(url, {
                method: 'POST',
                body: JSON.stringify({
                    prompt: prompt,
                    temperature: temperature
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
                    <span className="message">
                        {/* {jsx_msg.map((value, index) => {
                            return <Fragment key={index}>{value}<br/></Fragment>
                        })} */}
                    </span>
                    <span className="message">
                        Lorem ipsum dolor sit amet consectetur, adipisicing elit. Beatae suscipit nesciunt voluptatum enim eum consequuntur nihil ratione rem inventore? Soluta, possimus eum iste iusto quo provident quia laborum fugit rem.
                    </span>
                    <span className="message">
                        Lorem ipsum dolor sit amet consectetur adipisicing elit. Accusantium magni ratione, earum adipisci voluptatum excepturi accusamus provident similique doloremque minima natus deleniti harum. Voluptatum veniam eos inventore enim cum recusandae?
                    </span>
                    <span className="message">
                        Lorem ipsum dolor sit amet consectetur adipisicing elit. Accusantium magni ratione, earum adipisci voluptatum excepturi accusamus provident similique doloremque minima natus deleniti harum. Voluptatum veniam eos inventore enim cum recusandae?
                    </span>
                    <span className="message">
                        Lorem ipsum dolor sit amet consectetur adipisicing elit. Accusantium magni ratione, earum adipisci voluptatum excepturi accusamus provident similique doloremque minima natus deleniti harum. Voluptatum veniam eos inventore enim cum recusandae?
                    </span>
                </div>
                <div className="prompts-container">
                    <input onChange={(event) => setPrompt(event.target.value)} value={prompt} className="prompts-field" type="text" placeholder="Type a prompt e.g. Dostoevsky"/>
                    <button onClick={generate} className="generate-btn">generate</button>
                </div>
            </div>
        </div>
    );
}