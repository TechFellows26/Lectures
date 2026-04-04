(function() {
  var root = document.documentElement;
  var stored = localStorage.getItem('lebnet-theme');
  if (stored === 'light') root.classList.add('theme-light');

  document.addEventListener('DOMContentLoaded', function() {
    var btn = document.querySelector('.theme-toggle');
    if (btn) btn.addEventListener('click', function() {
      root.classList.toggle('theme-light');
      localStorage.setItem('lebnet-theme',
        root.classList.contains('theme-light') ? 'light' : 'dark');
    });
  });
})();
