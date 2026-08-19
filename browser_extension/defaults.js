/* One source of truth for the scroller and the tag filter: loaded by both the
 * content script and the popup, so neither can drift from the other.
 */

const SCROLL_DEFAULTS = {
  stepPx: 900, // fallback step when no next post is loaded yet
  minInterval: 2000,
  maxInterval: 4000,
  pauseAfter: 20,
  pauseSeconds: 30,
  maxScrolls: 300,
  minDuration: 5, // same window the downloader enforces later
  maxDuration: 60,
  animalTags:
    "animal, animals, pet, pets, dog, puppy, cat, kitten, kitty, hamster, " +
    "parrot, horse, goat, duck, panda, fox, raccoon, lama, llama, alpaca, " +
    "pig, piglet, животн, кот, кошка, котик, собак, пес, щенок, попугай, " +
    "хомяк, енот, питом, лама, свинь, свинк, порос",
  funnyTags:
    "funny, fun, lol, humor, comedy, cute, fail, смешн, юмор, прикол, мил",
};

// Rounds at the bottom of a feed that stopped growing before the scroller gives up.
const IDLE_ROUNDS_TO_STOP = 3;
