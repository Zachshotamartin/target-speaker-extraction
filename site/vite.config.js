import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
const proxy={'/api/one-voice':{target:'http://127.0.0.1:5294',changeOrigin:true,rewrite:path=>path.replace('/api/one-voice','')}};
export default defineConfig({plugins:[react()],server:{proxy},preview:{proxy}});
