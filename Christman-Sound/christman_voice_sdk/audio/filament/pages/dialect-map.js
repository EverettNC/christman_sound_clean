/** Filament dialect tour. Train walks one selected region at a time, out loud.
 * Kentucky → Tennessee → Texas → Louisiana Creole/bayou →
 * Florida cracker → Washington hippies → Montana → Utah →
 * Vegas (oreo / Chinese / Mandarin) → the rest of the map.
 */
(function (root) {
  const ZH = new Set(["ni-hao", "xie-xie", "hao", "ma", "dui", "mei-you", "dim sum", "siu mai", "cha siu", "lai see", "m'goi", "ho sik"]);
  const TH = new Set(["sawasdee", "khrap", "ka", "nam prik", "pad krapow", "sabai"]);

  const MAP = [
    { key: "kentucky", label: "Kentucky", lang: "en-US", tokens: ["holler", "y'uns", "nigh", "reckon", "poke", "yonder", "ain't", "mash", "cain't"] },
    { key: "tennessee", label: "Tennessee", lang: "en-US", tokens: ["holler", "reckon", "yonder", "ain't", "mash", "nigh", "y'uns", "cain't", "bless"] },
    { key: "texas", label: "Texas drawl", lang: "en-US", tokens: ["howdy", "y'all", "fixin'", "over yonder", "darlin'", "pardner", "cain't", "might could", "all hat", "ma'am"] },
    {
      key: "louisiana",
      label: "Louisiana Creole · bayou",
      lang: "en-US",
      tokens: ["cher", "sha", "lagniappe", "couyon", "envie", "boudin", "who dat", "yeah you right", "fais-dodo", "pirogue", "bayou", "gumbo", "zydeco", "ova", "dere", "dat", "dis"],
    },
    {
      key: "florida",
      label: "Florida cracker · hep · redneck",
      lang: "en-US",
      tokens: ["y'all", "ain't", "cain't", "fixin'", "yonder", "reckon", "might could", "palmetto", "gator", "hammock", "swamp", "hep", "cracker", "tain't", "sho nuff", "done told", "boy"],
    },
    {
      key: "washington",
      label: "Washington hippies",
      lang: "en-US",
      tokens: ["the mountain", "drizzle", "the pass", "spendy", "you bet", "copacetic", "far out", "rainier", "moss", "the island"],
    },
    {
      key: "montana",
      label: "Montana",
      lang: "en-US",
      tokens: ["big sky", "huckleberry", "the divide", "you bet", "range", "the pass", "butte", "howdy"],
    },
    {
      key: "utah",
      label: "Utah",
      lang: "en-US",
      tokens: ["the trail", "slickrock", "wash", "the lake", "ward", "pioneer", "mesa", "hoodoo", "desert"],
    },
    {
      key: "vegas",
      label: "Vegas · oreo · Chinese · Mandarin",
      lang: "en-US",
      tokens: [
        "the strip", "comped", "whale", "high roller", "tap out",
        "bruh", "on god", "no cap", "you feel me", "straight up", "fam",
        "dim sum", "siu mai", "cha siu", "lai see", "m'goi",
        "ni-hao", "xie-xie", "hao", "ma", "dui", "mei-you",
      ],
    },
    { key: "minnesota", label: "Minnesota", lang: "en-US", tokens: ["ope", "you betcha", "dontcha know", "uff-da", "hotdish", "yeah no", "no yeah"] },
    { key: "brooklyn", label: "Brooklyn", lang: "en-US", tokens: ["madonn'", "gabagool", "fuhgeddabou", "oobatz", "bodega", "deadass", "yerr", "brick", "son"] },
    { key: "bronx", label: "the Bronx", lang: "en-US", tokens: ["deadass", "bodega", "wallahi", "ock", "nahmean", "brick", "son", "fam", "good looks", "yerr"] },
    { key: "caribbean", label: "Caribbean US", lang: "en-US", tokens: ["irie", "wagwan", "likkle", "yute", "mon", "ting", "bruh"] },
    { key: "hawaiian", label: "Hawaiian Pidgin", lang: "en-US", tokens: ["da kine", "howzit", "pau", "mahalo", "bumbai"] },
    { key: "thai", label: "Thai", lang: "th-TH", tokens: ["sawasdee", "khrap", "ka", "nam prik", "pad krapow", "sabai"] },
  ];

  const TOUR = ["kentucky", "tennessee", "texas", "louisiana", "florida", "washington", "montana", "utah", "vegas"];
  let enabled = TOUR.slice();
  let i = 0;
  let queue = [];
  let current = MAP[0];
  let talking = false;
  let talkTimer = 0;

  function pool() {
    const on = MAP.filter((m) => enabled.includes(m.key));
    return on.length ? on : MAP;
  }

  function nextPack() {
    const p = pool();
    current = p[i % p.length];
    i += 1;
    queue = current.tokens.slice();
    root.filamentRegion = current.label;
    root.filamentPack = current;
    return current;
  }

  function langFor(tok) {
    if (ZH.has(tok)) return "zh-CN";
    if (TH.has(tok)) return "th-TH";
    return (current && current.lang) || "en-US";
  }

  function pickVoice(lang) {
    const vs = (root.speechSynthesis && root.speechSynthesis.getVoices()) || [];
    const want = (lang || "en-US").toLowerCase();
    return (
      vs.find((v) => v.lang.toLowerCase() === want) ||
      vs.find((v) => v.lang.toLowerCase().startsWith(want.slice(0, 2))) ||
      vs.find((v) => v.lang.toLowerCase().startsWith("en")) ||
      null
    );
  }

  function speak(tok, lang) {
    talking = true;
    if (talkTimer) clearTimeout(talkTimer);
    const done = () => {
      talking = false;
    };
    talkTimer = setTimeout(done, Math.max(700, 2800 / (Number(root.filamentSpeed) || 1)));
    if (root.filamentMute || !tok || !root.speechSynthesis) return;
    const u = new SpeechSynthesisUtterance(tok);
    u.lang = lang || langFor(tok);
    u.rate = 0.88;
    u.pitch = 1;
    const voice = pickVoice(u.lang);
    if (voice) u.voice = voice;
    u.onend = done;
    u.onerror = done;
    try {
      root.speechSynthesis.cancel();
      root.speechSynthesis.speak(u);
    } catch (_err) {
      done();
    }
  }

  function nextToken() {
    if (!queue.length) nextPack();
    const tok = queue.shift();
    root.filamentToken = tok;
    root.filamentLang = langFor(tok);
    return tok;
  }

  function setEnabled(keys) {
    const next = (keys || []).filter((k) => MAP.some((m) => m.key === k));
    enabled = next.length ? next : TOUR.slice();
    i = 0;
    queue = [];
    nextPack();
  }

  function busy() {
    return talking || !!(root.speechSynthesis && root.speechSynthesis.speaking);
  }

  nextPack();
  root.FILAMENT_MAP = MAP;
  root.FILAMENT_TOUR = TOUR;
  root.FILAMENT_NEXT_PACK = nextPack;
  root.FILAMENT_NEXT_TOKEN = nextToken;
  root.FILAMENT_SPEAK = speak;
  root.FILAMENT_SET_ENABLED = setEnabled;
  root.FILAMENT_BUSY = busy;
  root.FILAMENT_LANG_FOR = langFor;
  root.filamentRegion = current.label;
  root.filamentSpeed = 1;
  function gap() {
    const s = Number(root.filamentSpeed) || 1;
    return Math.max(220, 1200 / s);
  }
  root.FILAMENT_GAP = gap;
  if (root.speechSynthesis && root.speechSynthesis.getVoices) {
    root.speechSynthesis.getVoices();
    root.speechSynthesis.addEventListener("voiceschanged", () => root.speechSynthesis.getVoices());
  }
})(window);
