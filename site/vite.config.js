import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
const proxy={'/api/one-voice':{target:'http://127.0.0.1:5294',changeOrigin:true,rewrite:path=>path.replace('/api/one-voice','')},'/api/poc':{target:'http://127.0.0.1:5296',changeOrigin:true,rewrite:path=>path.replace('/api/poc','')}};
export default defineConfig({plugins:[react()],server:{proxy},preview:{proxy}});
