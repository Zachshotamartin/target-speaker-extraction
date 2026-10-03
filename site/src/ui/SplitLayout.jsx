import React from 'react';
import './ui.css';
export default function SplitLayout({sidebar, children, className = ''}) {
  return <div className={`ui-split-layout ${className}`}><aside className="ui-split-sidebar">{sidebar}</aside><div className="ui-split-main">{children}</div></div>;
}
