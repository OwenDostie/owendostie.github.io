
//// TIMELINE ////
//todo:

// ADD FADING BEHAVIOR
focusTimeline = () =>{
    const about = document.getElementById('about');
    const timeline = document.getElementById('timeline');
    const about_text = about.getElementsByClassName("about__text")[0];
    const currentScrollPosition = window.pageYOffset;
    const elementOffsetTop = document.getElementById('about').offsetTop;
    const scrollDiff = currentScrollPosition - elementOffsetTop;

    if (scrollDiff > 0) {
        about.style.opacity = Math.max(0, Math.min(1, 1-scrollDiff/(about.getBoundingClientRect().height*1/3)));
        timeline.style.opacity = 1 - Math.max(0, Math.min(1, 1-scrollDiff/(about.getBoundingClientRect().height*2/3)));
    }
    else {
        about.style.opacity = 1
        timeline.style.opacity = 0;
    }
}

jumpToTimeline = () =>{
    const timeline = document.getElementById('timeline');

    window.scroll({
        top: timeline.offsetTop,
        left: 0,
        behavior: 'smooth'
    });
}



//// INTERACTIVE TIMELINE LINE ////
// todo:
// - update value when scrolling


showTimelineMarker = () => {
    if (bgFade.style.opacity == 0) {
    let tllm = document.getElementById('timeline__line--marker');
    let tllmt = document.getElementById('timeline__line--marker-text');
    let rect=tll.getBoundingClientRect();
    let markerPos = Math.max(Math.min(event.pageY - window.pageYOffset,rect.bottom), rect.top);
    tllm.style.top=String(markerPos)+'px'; 
    tllm.style.left=String(rect.left+1)+'px';
    tllm.style.opacity=1;
    tllmt.style.top=String(markerPos)+'px';
    tllmt.style.left=String(rect.left-4)+'px';
    let markerDate=new Date((rect.height-(markerPos-rect.top))/tlScale+timelineStart);
    tllmt.innerHTML=markerDate.toLocaleString('default', { month: 'long' })+' '+markerDate.getFullYear();
    tllmt.style.opacity=1;
    }
}

hideTimelineMarker = () => {
    let tll = document.getElementById('timeline__line');
    let tllm = document.getElementById('timeline__line--marker');
    let tllmt = document.getElementById('timeline__line--marker-text');
    tllm.style.opacity=0; tllmt.style.opacity=0;
}

let tll = document.getElementById('timeline__line');
let tllm = document.getElementById('timeline__line--marker');
let tllmt = document.getElementById('timeline__line--marker-text');
tll.addEventListener('mousemove', showTimelineMarker);
tll.addEventListener('mouseleave', hideTimelineMarker);
tllm.addEventListener('mousemove', showTimelineMarker);
tllm.addEventListener('mouseleave', hideTimelineMarker);
tllmt.addEventListener('mousemove', showTimelineMarker);
tllmt.addEventListener('mouseleave', hideTimelineMarker);


//// TIMELINE ITEMS ////


const about = document.getElementById('about');
const rect = about.getBoundingClientRect();
const vpheight = window.innerHeight;
about.style.height = String(vpheight - rect.top)+'px';

window.addEventListener('scroll', focusTimeline);
document.getElementsByClassName("about__arrowdown")[0].addEventListener('click', jumpToTimeline);

// Get current date
const today = new Date().getTime();
const timelineStart = new Date(2016, 8, 1).getTime();
const timelineHeight = document.getElementsByClassName('timeline__body')[0].clientHeight;
const tlScale = 1/(today - timelineStart)*timelineHeight

// Add colors and organize timeline
const colors = ['crimson','cadetblue','darkseagreen','purple','slateblue','forestgreen']
var items = document.querySelectorAll(".timeline__item");
for(var i=0; i<items.length; i++){

    // items[i].style.color = colors[i % colors.length]
        // items[i].children.getElementsByClassName('timeline__item--marker').style.borderColor = colors[i % colors.length]

    // If the item has a start date position vertically accordingly
    if (typeof items[i].dataset.begin !== 'undefined') {
        var dateEnd, dateBegin = 0
        if (items[i].dataset.end == 'present') {
            dateEnd = today;
        }
        else {
            dateEnd = Math.min(Date.parse(items[i].dataset.end), today);
        }
        dateBegin = Math.max(Date.parse(items[i].dataset.begin), timelineStart);

        items[i].style.top = String(Math.round((today-dateEnd)*tlScale)) + 'px'
        items[i].style.height = String(Math.round((dateEnd - dateBegin)*tlScale)) + 'px'
    }

}


//// ADD INTERACTION TO TIMELINE ITEM ////


const bgFade = document.getElementById("bg-fade");
const tl_line = document.getElementById('timeline__line')
items = document.querySelectorAll(".timeline__item--label");
for(var i=0; i<items.length; i++){
    items[i].onmouseover = function(event){
        if (event.currentTarget.classList.contains("timeline__item--description")) {
            this.parentElement.classList.remove("timeline__item--hover");
        } else {this.parentElement.classList.add("timeline__item--hover");}
        // console.log(event.target.classList);
    }
    items[i].onmouseleave = function(){
        this.parentElement.classList.remove("timeline__item--hover");
    }
    items[i].addEventListener('mousedown',function(event){
        var marker = this.parentElement.getElementsByClassName("timeline__item--marker")[0]
        if (this.parentElement.classList.contains("timeline__item--focused")) {
            this.parentElement.classList.remove("timeline__item--focused");
            marker.style.transform = 'translateX(0px)'
            bgFade.classList.remove("bg-fade--visible");
        } else {
            this.parentElement.classList.add("timeline__item--focused");
            marker.style.transform = `translateX(${(tl_line.getBoundingClientRect().x - marker.getBoundingClientRect().x) - 2}px)`
            bgFade.classList.add("bg-fade--visible");
        }
        event.stopPropagation();
    })
    // DIFFERENT BEHAVIOR WHEN CLICKING DESCRIPTIVE TEXT
    // items[i].getElementsByClassName("timeline__item--description")[0].addEventListener('click',function(event){
    // })
}


items = document.querySelectorAll(".timeline__item--description");
for(var i=0; i<items.length; i++){
    items[i].addEventListener("mouseover", e =>{
        e.currentTarget.parentElement.parentElement.classList.remove("timeline__item--hover");
        e.stopPropagation();
        // console.log(e.currentTarget.parentElement.parentElement);
        // console.log(e.target.parentElemennt);
        // this.parentElement.classList.remove("timeline__item--hover");
        // console.log(event.target.classList);
    })
    items[i].addEventListener("mousedown", e => {
        e.stopPropagation();
        // console.log(event.target.classList);
    })
}

// Remove focus when clicking away from an item
document.addEventListener("mousedown",function(event){
    items = document.querySelectorAll(".timeline__item");
    bgFade.classList.remove("bg-fade--visible");
    for (var i=0; i<items.length; i++){
        items[i].classList.remove("timeline__item--focused");
        items[i].getElementsByClassName("timeline__item--marker")[0].style.transform = 'translateX(0px)';
    }
})