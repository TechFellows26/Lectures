(function() {
  var canvas, ctx, w, h, cols, rows;
  var mx = -1000, my = -1000;
  var gap = 24, radius = 1.2, glowRadius = 120;
  var waveOffset = 0;
  var accent;

  function hexToRgb(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    return {
      r: parseInt(hex.substring(0, 2), 16),
      g: parseInt(hex.substring(2, 4), 16),
      b: parseInt(hex.substring(4, 6), 16)
    };
  }

  function readAccent() {
    accent = hexToRgb(
      getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#7cb87c'
    );
  }

  function init() {
    canvas = document.getElementById('dots-bg');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    readAccent();
    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', function(e) {
      mx = e.clientX;
      my = e.clientY;
    });
    window.addEventListener('mouseout', function() {
      mx = -1000; my = -1000;
    });
    requestAnimationFrame(loop);
  }

  function resize() {
    var dpr = window.devicePixelRatio || 1;
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(w / gap) + 1;
    rows = Math.ceil(h / gap) + 1;
    readAccent();
  }

  function loop() {
    draw();
    requestAnimationFrame(loop);
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    waveOffset += 0.008;

    var r_c = accent.r, g_c = accent.g, b_c = accent.b;

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var x = c * gap;
        var y = r * gap;

        var dx = mx - x;
        var dy = my - y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        var proximity = Math.max(0, 1 - dist / glowRadius);

        var wave = Math.sin(c * 0.15 + r * 0.15 + waveOffset) * 0.5 + 0.5;
        var baseAlpha = 0.10 + wave * 0.06;
        var alpha = baseAlpha + proximity * 0.35;
        var size = radius + proximity * 1.2 + wave * 0.3;

        ctx.beginPath();
        ctx.arc(x, y, size, 0, 6.2832);
        ctx.fillStyle = 'rgba(' + r_c + ',' + g_c + ',' + b_c + ',' + alpha + ')';
        ctx.fill();
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
