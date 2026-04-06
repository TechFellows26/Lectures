(function() {
  function cleanHeadingText(el) {
    var clone = el.cloneNode(true);
    var mathml = clone.querySelectorAll('.katex-mathml');
    mathml.forEach(function(m) { m.remove(); });
    return clone.textContent.replace(/\s+/g, ' ').trim();
  }

  function buildSidebar() {
    var sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    document.body.classList.add('has-sidebar');

    var nav = sidebar.querySelector('.sidebar-nav');
    if (!nav) return;

    nav.innerHTML = '';

    var headings = document.querySelectorAll('.paper h2, .paper h3');
    if (!headings.length) return;

    headings.forEach(function(h, i) {
      if (!h.id) h.id = 'section-' + i;
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = cleanHeadingText(h);
      if (h.tagName === 'H3') a.classList.add('sub');
      a.addEventListener('click', function(e) {
        e.preventDefault();
        h.scrollIntoView({ behavior: 'smooth' });
        history.replaceState(null, '', '#' + h.id);
      });
      li.appendChild(a);
      nav.appendChild(li);
    });

    var links = nav.querySelectorAll('a');
    var ticking = false;

    function updateActive() {
      var scrollY = window.scrollY + 100;
      var current = null;
      headings.forEach(function(h) {
        if (h.offsetTop <= scrollY) current = h.id;
      });
      links.forEach(function(a) {
        var isActive = a.getAttribute('href') === '#' + current;
        a.classList.toggle('active', isActive);
        if (isActive) {
          var linkTop = a.offsetTop;
          var sidebarHeight = sidebar.clientHeight;
          sidebar.scrollTo({ top: linkTop - sidebarHeight / 3, behavior: 'smooth' });
        }
      });
      ticking = false;
    }

    window.addEventListener('scroll', function() {
      if (!ticking) { ticking = true; requestAnimationFrame(updateActive); }
    });
    updateActive();

    var toggle = document.querySelector('.sidebar-toggle');
    if (toggle) {
      var collapsed = localStorage.getItem('lebnet-sidebar') === 'collapsed';
      if (collapsed) document.body.classList.add('sidebar-collapsed');

      toggle.addEventListener('click', function() {
        var isCollapsed = document.body.classList.toggle('sidebar-collapsed');
        localStorage.setItem('lebnet-sidebar', isCollapsed ? 'collapsed' : 'open');
      });
    }
  }

  if (document.readyState === 'complete') {
    buildSidebar();
  } else {
    window.addEventListener('load', buildSidebar);
  }

  document.addEventListener('DOMContentLoaded', function() {
    var sidebar = document.getElementById('sidebar');
    if (sidebar) {
      document.body.classList.add('has-sidebar');
      if (localStorage.getItem('lebnet-sidebar') === 'collapsed') {
        document.body.classList.add('sidebar-collapsed');
      }
    }
  });
})();
