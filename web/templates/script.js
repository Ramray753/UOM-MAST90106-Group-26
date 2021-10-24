// const dropArea = document.querySelector(".drag-area"),
// dragText = dropArea.querySelector("header"),
// button = dropArea.querySelector("button"),
// input = dropArea.querySelector("input");
//
// let file;
//
// button.onclick = () =>{
//     input.click();
// }
//
// input.addEventListener("change", function () {
//     file = this.files[0];
//     showFile()
//
// })
//
// dropArea.addEventListener("dragover", ()=>{
//     event.preventDefault();
//     dropArea.classList.add("active");
// })
// dropArea.addEventListener("dragleave", ()=>{
//     event.preventDefault();
//     dropArea.classList.add("active");
// })
//
// dropArea.addEventListener("drop", ()=>{
//     event.preventDefault();
//     file = event.dataTransfer.files[0];
//     showFile()
//
// })
//
// function showFile() {
//     let fileType = file.type;
//     console.log(file)
//
//     let validExtensions = ["csv", "csv", "csv"];
//     if(validExtensions.includes(fileType)){
//         let fileReader = new FileReader();
//         fileReader.onload = ()=>{
//             let fileURL = fileReader.result;
//         }
//         fileReader.readAsDataURL(file);
//     }else{
//         alert("This is not a CSV file!");
//         dropArea.classList.remove("active");
//     }
// }

document.querySelectorAll(".drop-zone__input").forEach(inputElement => {
    const dropZoneElement = inputElement.closest(".drop-zone");

    dropZoneElement.addEventListener("click", e =>{
        inputElement.click();
    })

    inputElement.addEventListener("change", e => {
        if (inputElement.files.length){
            updataThumbnail(dropZoneElement, inputElement.files[0]);
        }
    })

    dropZoneElement.addEventListener("dragover", e=>{
        e.preventDefault();
        dropZoneElement.classList.add("drop-zone--over");

    });
    ['dragleave','dragend'].forEach(type =>{
        dropZoneElement.addEventListener(type, e =>{
            dropZoneElement.classList.remove("drop-zone--over");
        });

    });

    dropZoneElement.addEventListener("drop",e=>{
        e.preventDefault();
        if (e.dataTransfer.files.length){
            inputElement.files = e.dataTransfer.files;
            updataThumbnail(dropZoneElement,e.dataTransfer.files[0]);
        }
        dropZoneElement.classList.remove("drop-zone--over");
        
    });

});

@param {HTMLElement} dropZoneElement

@param {File} file

function updataThumbnail(dropZoneElement,file) {
    let thumbnailElement = dropZoneElement.querySelector(".drop-zone__thumb");

    console.log(file);

    if (dropZoneElement.querySelector(".drop-zone__prompt")){
        dropZoneElement.querySelector(".drop-zone__prompt").remove();
    }

    if (!thumbnailElement) {
        thumbnailElement = document.createElement("div");
        thumbnailElement.classList.add("drop=zone__thumb");
        dropZoneElement.appendChild(thumbnailElement);
    }

    thumbnailElement.dataset.label = file.name;


    if (file.type.startsWith("image/")){
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () =>{
          thumbnailElement.style.backgroundImage = "url('${ reader.result}')";
        };
    } else{
        thumbnailElement.style.backgroundImage = null;
    }
}
