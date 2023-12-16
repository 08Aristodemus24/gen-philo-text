export default function Message({ message: my_message }){
    // create the compnent that will contain all the
    // messages derived from messages state, by splitting
    // the message into individual characters for css animations
    const lines = my_message['message'].split('\n').map((value, index) => {
        return <li className="line" key={index}>{value}</li>
    });
    
    return <ul className="message">
        {lines}
    </ul>
}