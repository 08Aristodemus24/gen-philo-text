import { Fragment, useState } from "react";



export default function Glass(){
    const test_msg = `hello world\nmy name is michael, Lorem ipsum dolor sit amet consectetur adipisicing elit. Porro sapiente odio\n accusamus ad ipsam incidunt veniam adipisci, ipsum beatae, possimus \nnemo est fugit ullam, numquam nisi amet fuga vel aut!`;
    const jsx_msg = test_msg.split('\n');
    console.log(jsx_msg)
    return (
        <div className="glass-container">
            <div className="glass">
                <div className="messages-container">
                    <span className="message">
                        {jsx_msg.map((value, index) => {
                            return <Fragment key={index}>{value}<br/></Fragment>
                        })}
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
                    <input className="prompts-field" type="text" placeholder="Type a prompt e.g. Dostoevsky"/>
                    <button className="generate-btn">generate</button>
                </div>
            </div>
        </div>
    );
}