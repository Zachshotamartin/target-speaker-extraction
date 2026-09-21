import React from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.jsx';
import './product.css';
import './design-tokens.css';
import './workspace-shell.css';
import './header.css';
import './motion.css';

createRoot(document.getElementById('root')).render(<App/>);
