$(document).ready(function () {
    let faceBookHeader = $('ul').offset().top;
    $(window).on('scroll',function(e){
        let scrollTop = $(this).scrollTop()
        console.log(scrollTop,faceBookHeader,'a')//콘솔로 위치 비교, 작동여부
        if(scrollTop > faceBookHeader){/////스크롤 위치 비교후 작동
            console.log('b')//콘솔로 작동여부
            $('ul').css({
                'position':'fixed',
                'top': 0
            })
        }
        else{////스크롤 위치 비교후 원상태로
            console.log('c')//콘솔로 작동여부
            $('ul').css('position','relative');
        };

    });
})