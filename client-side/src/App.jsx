import { useEffect, useState } from 'react'
import Section from './components/Section';

import './content.css'
import Glass from './components/Glass';

function App() {
  useEffect(() => {
    console.log('fetching resources');
    // const url = 'http://127.0.0.1:5000/send-mail';
    // const response = fetch(url);
  });

  let messages, setMessages = useState(null);

  return (
    <Section section-name={"landing"}>
      <Glass></Glass>
    </Section>
  )
}

export default App
