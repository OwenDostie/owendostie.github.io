
//// INTERACTIVE TIMELINE LINE ////
// todo:
// - update value when scrolling
const tll = document.getElementById('timeline__line');
const tllm = document.getElementById('timeline__line--marker');
const tllmt = document.getElementById('timeline__line--marker-text');
showTimelineMarker = () => {
    if (bgFade.style.opacity == 0) {
    const tllm = document.getElementById('timeline__line--marker');
    const tllmt = document.getElementById('timeline__line--marker-text');
    const rect=tll.getBoundingClientRect();
    const markerPos = Math.max(Math.min(event.pageY - window.pageYOffset,rect.bottom), rect.top);
    tllm.style.top=String(markerPos)+'px'; 
    tllm.style.left=String(rect.left+1)+'px';
    tllm.style.opacity=1;
    tllmt.style.top=String(markerPos)+'px';
    tllmt.style.left=String(rect.left-4)+'px';
    const markerDate=new Date((rect.height-(markerPos-rect.top))/tlScale+timelineStart);
    tllmt.innerHTML=markerDate.toLocaleString('default', { month: 'long' })+' '+markerDate.getFullYear();
    tllmt.style.opacity=1;
    }
}

hideTimelineMarker = () => {
    const tll = document.getElementById('timeline__line');
    const tllm = document.getElementById('timeline__line--marker');
    const tllmt = document.getElementById('timeline__line--marker-text');
    tllm.style.opacity=0; tllmt.style.opacity=0;
}

tll.addEventListener('mousemove', showTimelineMarker);
tll.addEventListener('mouseleave', hideTimelineMarker);
tllm.addEventListener('mousemove', showTimelineMarker);
tllm.addEventListener('mouseleave', hideTimelineMarker);
tllmt.addEventListener('mousemove', showTimelineMarker);
tllmt.addEventListener('mouseleave', hideTimelineMarker);