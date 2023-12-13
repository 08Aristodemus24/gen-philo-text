import { useEffect, useState } from 'react'
import Section from './components/Section';

import './content.css'

function App() {
  useEffect(() => {
    console.log('fetching resources');
  })

  return (
    <Section section-name={"landing"}>
      <h1>hello world</h1>
      <button>test</button>
    </Section>
  )
}

export default App
