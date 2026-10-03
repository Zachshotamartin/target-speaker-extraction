import React, {useEffect, useState} from 'react';
export default function WorkspaceFeedback({message}) {
  const [dismissed, setDismissed] = useState(false);
  useEffect(() => {setDismissed(false); const timer = setTimeout(() => setDismissed(true), 5000); return () => clearTimeout(timer);}, [message]);
  if (!message || dismissed) return null;
  return <div className="workspace-feedback" role="status"><span>{message}</span><button type="button" aria-label="Dismiss notification" onClick={() => setDismissed(true)}>×</button></div>;
}
