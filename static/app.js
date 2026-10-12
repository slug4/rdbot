const form=document.getElementById('ask'),button=document.getElementById('submit'),result=document.getElementById('result');
function element(tag,text,className){const node=document.createElement(tag);node.textContent=text;if(className)node.className=className;return node;}
form.addEventListener('submit',async event=>{
 event.preventDefault();button.disabled=true;button.textContent='Finding supporting sources…';result.replaceChildren();
 try{
  const response=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('question').value})});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not complete this question.');
  result.append(element('span',data.mode+' retrieval','badge'),element('p',data.answer,'answer-text'));
  if(data.partial_sources)result.append(element('p','Some source pages were unavailable. These results cover only the pages successfully fetched.','partial'));
  for(const source of data.sources){const box=element('div','','source'),link=element('a',`[Source ${source.id}] ${source.url}`);link.href=source.url;link.target='_blank';link.rel='noopener noreferrer';box.append(link,element('p',source.text));result.append(box);}
 }catch(error){result.append(element('p',error.message,'error'));}finally{button.disabled=false;button.textContent='Find an answer ↗';}
});
