import { useEffect, useState } from 'react'
import Section from './components/Section';

import './content.css'
import Glass from './components/Glass';

function App() {
  useEffect(() => {
    console.log('fetching resources');
  });

  return (
    <Section section-name={"landing"}>
      <Glass></Glass>
    </Section>
  )
}

export default App
