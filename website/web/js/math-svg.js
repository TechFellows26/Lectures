(function () {
  var macros = {
    '\\R': '\\mathbb{R}', '\\E': '\\mathbb{E}',
    '\\Var': '\\text{Var}', '\\Cov': '\\text{Cov}',
    '\\Bias': '\\text{Bias}', '\\MSE': '\\text{MSE}',
    '\\tr': '\\text{tr}', '\\rank': '\\text{rank}',
    '\\N': '\\mathbb{N}', '\\Z': '\\mathbb{Z}'
  };

  var delimiters = [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false }
  ];

  function process() {
    if (typeof renderMathInElement === 'undefined') {
      setTimeout(process, 50);
      return;
    }

    var texts = Array.prototype.slice.call(document.querySelectorAll('svg text'));

    texts.forEach(function (textEl) {
      if (textEl.textContent.indexOf('$') === -1) return;

      var raw = textEl.textContent;
      var x = parseFloat(textEl.getAttribute('x') || 0);
      var y = parseFloat(textEl.getAttribute('y') || 0);
      var fontSize = parseFloat(textEl.getAttribute('font-size') || 12);
      var fill = textEl.getAttribute('fill') || 'currentColor';
      var fontWeight = textEl.getAttribute('font-weight') || 'normal';
      var anchor = textEl.getAttribute('text-anchor') || 'start';

      var ns = 'http://www.w3.org/2000/svg';
      var fo = document.createElementNS(ns, 'foreignObject');

      var w = 300;
      var h = fontSize * 4;
      var foX = x;
      if (anchor === 'middle') foX = x - w / 2;
      else if (anchor === 'end') foX = x - w;

      fo.setAttribute('x', foX);
      fo.setAttribute('y', y - fontSize * 1.2);
      fo.setAttribute('width', w);
      fo.setAttribute('height', h);
      fo.style.overflow = 'visible';
      fo.style.pointerEvents = 'none';

      var div = document.createElement('div');
      div.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
      div.style.fontSize = fontSize + 'px';
      div.style.color = fill;
      div.style.fontWeight = fontWeight;
      div.style.whiteSpace = 'nowrap';
      div.style.lineHeight = '1.2';
      div.style.fontFamily = "'Crimson Pro', Georgia, serif";
      if (anchor === 'middle') div.style.textAlign = 'center';
      else if (anchor === 'end') div.style.textAlign = 'right';

      div.textContent = raw;
      fo.appendChild(div);

      textEl.parentNode.insertBefore(fo, textEl);
      textEl.remove();

      renderMathInElement(fo, {
        delimiters: delimiters,
        macros: macros,
        throwOnError: false
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      setTimeout(process, 100);
    });
  } else {
    setTimeout(process, 100);
  }
})();
