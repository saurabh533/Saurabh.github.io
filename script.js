// You can add JavaScript functionality here
document.addEventListener("DOMContentLoaded", function () {
    // Example: Change the background color when a button is clicked
    const changeColorButton = document.getElementById("changeColorButton");
    const body = document.body;

    changeColorButton.addEventListener("click", function () {
        body.style.backgroundColor = getRandomColor();
    });

    // Function to generate a random color
    function getRandomColor() {
        const letters = "0123456789ABCDEF";
        let color = "#";
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }
});
