const form=document.getElementById('ask'),button=document.getElementById('submit'),result=document.getElementById('result'),question=document.getElementById('question');
const faqButtons=[...document.querySelectorAll('.faq-question')];
function element(tag,text,className){const node=document.createElement(tag);node.textContent=text;if(className)node.className=className;return node;}
async function ask(faqId){
 if(!question.value.trim())return;
 button.disabled=true;faqButtons.forEach(b=>b.disabled=true);button.textContent='Finding your answer…';result.replaceChildren();
 try{
  const response=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.value})});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not complete this question.');
  const label=data.mode==='safety'?'Medical advice requires a clinician':'Answer from RxDx sources';
  result.append(element('span',label,'badge'),element('p',data.answer,'answer-text'));
  if(data.partial_sources)result.append(element('p','Some source pages were unavailable. Please confirm details directly with RxDx.','partial'));
  for(const source of data.sources){
   const box=element('details','','source'),title=element('summary','Official source'),link=element('a',source.url);
   link.href=source.url;link.target='_blank';link.rel='noopener noreferrer';box.append(title,link,element('p',source.text));result.append(box);
  }
 }catch(error){result.append(element('p',error.message,'error'));}finally{button.disabled=false;faqButtons.forEach(b=>b.disabled=false);button.textContent='Get a simple answer ↗';}
}
form.addEventListener('submit',event=>{event.preventDefault();ask();});
faqButtons.forEach(b=>b.addEventListener('click',()=>{question.value=b.textContent;ask(b.dataset.id);if(window.innerWidth<760)document.querySelector('.conversation').scrollIntoView({behavior:'smooth'});}));
document.getElementById('faq-search').addEventListener('input',event=>{
 const term=event.target.value.toLowerCase().trim();let count=0;
 document.querySelectorAll('.faq-group').forEach(group=>{
  let shown=0;group.querySelectorAll('.faq-question').forEach(b=>{b.hidden=!b.textContent.toLowerCase().includes(term);if(!b.hidden)shown++;});
  group.hidden=shown===0;if(term&&shown)group.open=true;count+=shown;
 });
 document.getElementById('no-questions').hidden=count>0;
});
