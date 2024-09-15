$(document).ready(function () {
    function gugu(index){
        let contents = '';
        let parent = $('#result');
        for(let i=1; i < 10; i++){
            contents += '<p>'+ index + '*' + i + '= ' + (i*index) + '</p>';
        }
        parent.html(contents);
    };
    $('button').on('click', function (e) {
        let index = $(this).val();
        gugu(index);
    })
})