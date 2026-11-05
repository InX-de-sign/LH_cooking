// Offline behavior checks; run with `node tests/test_browser.cjs`.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../chatbot/static/index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/\r\n/g, '\n');
let uuidSequence = 0;

function harness(options = {}) {
  const elements = new Map();
  const images = [], revoked = [], draws = [];
  const sockets = [], windowEvents = {}, clearedIntervals = [];
  function element(tag) {
    const result = {
      textContent: '', innerHTML: '', dataset: {}, checked: false, children: [],
      value: '', hidden: false, listeners: {},
      options: [], tagName: (tag || '').toUpperCase(),
      showModal() { this.open = true; }, close() { this.open = false; },
      classList: { add() {}, remove() {} },
      addEventListener(name, fn) { this.listeners[name] = fn; },
      removeAttribute(name) { delete this[name]; }, click() { this.clicked = true; },
      appendChild(child) { this.children.push(child); child.parent = this; if (child.tagName === 'OPTION') this.options.push(child); },
      append(...children) { this.children.push(...children); },
      prepend(...children) { this.children.unshift(...children); },
      replaceChildren(...children) { this.children = children; },
      querySelectorAll() { return []; }, remove() {
        if (this.parent) {
          this.parent.children = this.parent.children.filter(child => child !== this);
          this.parent.options = this.parent.options.filter(child => child !== this);
        }
      },
    };
    if (tag === 'canvas') {
      result.getContext = () => ({ fillRect() {}, drawImage(...args) { draws.push(args); } });
      result.toDataURL = () => options.canvasData || 'data:image/jpeg;base64,YWJj';
    }
    return result;
  }
  const timers = [];
  const speech = {
    queue: [], said: [], getVoices() { return []; },
    cancel() { this.queue = []; },
    speak(utterance) { this.queue.push(utterance); this.said.push(utterance.text); },
  };
  const context = {
    document: {
      visibilityState: 'visible', addEventListener() {}, createElement: element,
      getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    },
    speechSynthesis: speech,
    SpeechSynthesisUtterance: function(text) { this.text = text; },
    localStorage: { getItem() { return 'test'; } },
    crypto: { randomUUID() { return `test-${++uuidSequence}`; } }, navigator: {},
    location: { protocol: 'http:', host: 'localhost:8000' },
    fetch: async () => ({ json: async () => ({ recipes: options.recipes || [] }) }),
    addEventListener(name, fn) { windowEvents[name] = fn; },
    URL: class extends URL {
      static createObjectURL() { return 'blob:test-photo'; }
      static revokeObjectURL(url) { revoked.push(url); }
    },
    Image: class {
      constructor() { this.naturalWidth = 4000; this.naturalHeight = 3000; images.push(this); }
      set src(value) { if (!options.delayImage) Promise.resolve().then(() => this.onload()); }
    },
    WebSocket: class {
      static OPEN = 1;
      constructor(url) { this.url = url; this.readyState = 0; this.sent = []; sockets.push(this); }
      send(body) { this.sent.push(JSON.parse(body)); }
      close() { this.closed = true; this.onclose?.(); }
    },
    setTimeout(fn) { timers.push(fn); }, clearTimeout() {}, setInterval() {}, clearInterval(id) { clearedIntervals.push(id); },
  };
  context.window = context;
  // Expose existing functions and skip opening a real socket in this harness.
  const instrumented = source.replace(/\n  connect\(\);\n\}\)\(\);/,
    '\n  window.test = { state, els, handle, stopSpeaking, ask, attachPhoto, clearPhoto, openProfile, saveProfile, loadHistory, loadHistoryDetail, connect, newChat, renderCooking };\n})();');
  assert.notEqual(instrumented, source);
  vm.runInNewContext(instrumented, context);
  context.test.els.voiceOut.checked = true;
  return { ...context.test, speech, timers, images, revoked, draws, sockets, windowEvents, clearedIntervals };
}

async function main() {
{
  const h = harness(); h.state.onboarded = true;
  const preview = { recipe: 'Chicken Curry', step: null, total_steps: 4, serves: 2, time_minutes: 35,
    ingredients: ['300 grams chicken', '200 millilitres coconut milk', '<b>literal ingredient</b>'] };
  await h.renderCooking(preview);
  assert.equal(h.els.ingredientsPanel.hidden, false);
  assert.equal(h.els.ingredientsPanel.open, true);
  assert.deepEqual(h.els.ingredientsList.children.map(li => li.textContent), preview.ingredients);
  assert.equal(h.els.ingredientsList.children[2].innerHTML, '', 'ingredients use safe text rendering');
  assert.equal(h.els.recipeDetails.textContent, 'Serves 2 · About 35 min');
  assert.equal(h.els.startCooking.hidden, false);
  h.els.startCooking.listeners.click();
  assert.equal(JSON.parse(h.state.queue[0]).message, 'Start cooking');
  await h.renderCooking({ ...preview, step: 0, step_text: 'Heat oil.' });
  assert.equal(h.els.startCooking.hidden, true);
  assert.equal(h.els.ingredientsPanel.hidden, false, 'ingredients stay available during cooking');
  assert.equal(h.els.ingredientsPanel.open, false, 'ingredients collapse when the tutorial starts');
  assert.equal(h.els.stepText.textContent, 'Heat oil.');
  await h.renderCooking({ recipe: 'Banana Bread', step: null, total_steps: 4, ingredients: ['3 bananas'] });
  assert.equal(h.els.ingredientsPanel.open, true, 'switching recipes opens the new ingredient list');
  assert.equal(h.els.ingredientsList.children.length, 1);
  h.state.onboarded = false;
  await h.renderCooking({ recipe: null, step: null, total_steps: 0 });
  assert.equal(h.els.ingredientsPanel.hidden, true, 'a fresh chat clears the previous recipe preview');
  assert.equal(h.els.ingredientsList.children.length, 0);
  assert.equal(h.els.startCooking.hidden, true);
}

{
  const h = harness({ recipes: [
    { name: 'Chicken Curry', aliases: ['curry chicken'], ingredients: ['coconut milk'], time_minutes: 35 },
    { name: 'Spaghetti Bolognese', aliases: ['spag bol'], ingredients: ['minced beef'], time_minutes: 45 },
  ] });
  h.state.onboarded = true;
  await h.renderCooking({ recipe: null, step: null, total_steps: 0 });
  assert.equal(h.els.recipePicks.children.length, 2);
  assert.equal(h.els.recipeSearchLabel.hidden, false);
  for (const term of ['curry', 'curry chicken', 'coconut']) {
    h.els.recipeSearch.value = term; h.els.recipeSearch.listeners.input();
    assert.equal(h.els.recipePicks.children.length, 1, `search matches ${term}`);
    assert.equal(h.els.recipePicks.children[0].dataset.recipe, 'Chicken Curry');
  }
  h.els.recipeSearch.value = 'nothing matches'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.hidden, true);
  assert.match(h.els.recipeSearchStatus.textContent, /No recipes found/);
  h.els.recipeSearch.value = ''; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.children.length, 2, 'clearing search restores all recipes');
}

{
  const h = harness();
  h.els.handsFree.checked = true;
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'Use 2.' });
  assert.equal(h.speech.said.length, 0, 'partial decimal is not spoken');
  h.handle({ type: 'delta', text: '5 cups. Then stir' });
  assert.deepEqual(h.speech.said, ['Use 2.5 cups.'], 'first sentence speaks before final response');
  h.speech.queue.shift().onend();
  assert.equal(h.timers.length, 0, 'hands-free must wait for stream completion');
  h.handle({ type: 'response', message: 'Use 2.5 cups. Then stir' });
  assert.deepEqual(h.speech.said, ['Use 2.5 cups.', 'Then stir'], 'final response is not repeated');
  assert.equal(h.els.log.children.filter(x => x.className === 'msg coach').length, 1,
    'streamed and final events share a single reply bubble');
  h.speech.queue.shift().onend();
  assert.equal(h.timers.length, 1, 'hands-free resumes only after speech finishes');
}

{
  const h = harness();
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'First. ' });
  h.stopSpeaking();
  h.handle({ type: 'delta', text: 'Second. ' });
  h.handle({ type: 'response', message: 'First. Second.' });
  assert.deepEqual(h.speech.said, ['First.'], 'stopping speech mutes subsequent chunks of that reply');
  assert.equal(h.state.awaitingReply, false);
}

{
  const h = harness();
  h.els.voiceOut.checked = false;
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'Quiet. ' });
  h.handle({ type: 'response', message: 'Quiet.' });
  assert.equal(h.speech.said.length, 0, 'voice-off mode stays silent');
}

{
  const h = harness();
  h.handle({ type: 'thinking' });
  const messages = h.els.log.children.length;
  h.ask('next');
  assert.equal(h.els.log.children.length, messages + 1);
  assert.equal(h.els.log.children.at(-1).className, 'msg note', 'a second request waits for the first');
}

{
  const h = harness();
  h.handle({ type: 'response', message: 'Cook tofu.', sources: [
    { url: 'https://recipes.example/tofu', title: '<script>recipe</script>' },
    { url: 'javascript:alert(1)', title: 'Unsafe' },
  ] });
  const bubble = h.els.log.children.find(x => x.className === 'msg coach');
  const sources = bubble.children.at(-1);
  const links = sources.children.filter(x => x && x.href);
  assert.equal(links.length, 1, 'unsafe source links are discarded');
  assert.equal(links[0].textContent, '<script>recipe</script>', 'source titles are rendered as text');
  assert.equal(links[0].rel, 'noopener noreferrer');
  assert.deepEqual(h.speech.said, ['Cook tofu.'], 'source URLs are not spoken');
}

{
  const h = harness();
  h.state.onboarded = true;
  h.state.visionAvailable = true;
  h.els.uploadPhoto.listeners.click();
  assert.equal(h.els.photoInput.clicked, true, 'paperclip opens the photo picker');
  await h.attachPhoto({ name: 'dinner.jpg', type: 'image/jpeg', size: 5000 });
  assert.equal(h.state.queue.length, 0, 'selecting a photo does not send it');
  assert.equal(h.els.attachment.hidden, false);
  assert.equal(h.els.photoPreview.hidden, false);
  assert.equal(h.draws[0][3], 1280, 'large photo is resized before attachment');
  assert.equal(h.draws[0][4], 960);
  assert.equal(h.revoked.length, 1, 'temporary object URL is released');
  const photo = h.state.photo;
  await h.attachPhoto(null);
  assert.equal(h.state.photo, photo, 'cancelling the picker preserves the attachment');
  h.state.audioUnlocked = true;
  assert.equal(h.ask('What ingredients are here?'), true);
  const sent = JSON.parse(h.state.queue[0]);
  assert.equal(sent.image, photo.data);
  assert.equal(sent.image_source, 'upload');
  assert.equal(sent.message, 'What ingredients are here?');
  assert.equal(h.state.photo, null, 'photo clears after sending');
  assert.equal(h.els.attachment.hidden, true);
  h.state.awaitingReply = false;
  h.ask('next');
  assert.equal(JSON.parse(h.state.queue[1]).image, undefined, 'photo is not resent with the next turn');
}

{
  const h = harness();
  h.state.visionAvailable = true;
  await h.attachPhoto({ name: 'dinner.png', type: 'image/png', size: 5000 });
  h.state.awaitingReply = true;
  h.els.textInput.value = 'Keep my draft';
  h.els.textForm.listeners.submit({ preventDefault() {} });
  assert.equal(h.els.textInput.value, 'Keep my draft', 'busy submission preserves text');
  assert.ok(h.state.photo, 'busy submission preserves the photo');
  h.state.awaitingReply = false;
  h.state.visionAvailable = false;
  assert.equal(h.ask('Analyze'), false, 'no silent send without a vision model');
  assert.ok(h.state.photo);
  h.state.visionAvailable = true;
  h.state.audioUnlocked = true;
  h.ask('');
  assert.equal(JSON.parse(h.state.queue[0]).message, 'What can you see in this photo?', 'photo-only send has a useful prompt');
}

{
  const h = harness({ delayImage: true });
  const pending = h.attachPhoto({ name: 'dinner.webp', type: 'image/webp', size: 5000 });
  assert.equal(h.ask('Wait for photo'), false, 'sending waits for image preparation');
  h.els.removePhoto.listeners.click();
  h.images[0].onload();
  await pending;
  assert.equal(h.state.photo, null, 'removing a photo cancels an in-flight decode');
  assert.equal(h.els.attachment.hidden, true);
  assert.equal(h.revoked.length, 1);
  await h.attachPhoto({ name: 'movie.gif', type: 'image/gif', size: 10 });
  assert.equal(h.state.photo, null);
  assert.ok(h.els.log.children.at(-1).textContent.includes('JPEG, PNG or WebP'));
  await h.attachPhoto({ name: 'huge.jpg', type: 'image/jpeg', size: 20_000_001 });
  assert.ok(h.els.log.children.at(-1).textContent.includes('20 MB'));
}

{
  const h = harness();
  const requests = [];
  h.state.ws = { readyState: 1, send(body) { requests.push(JSON.parse(body)); } };
  const opening = h.openProfile();
  assert.equal(h.els.profileDialog.open, true);
  assert.equal(h.els.saveProfile.disabled, true, 'cannot save before the profile loads');
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id,
    profile: { cook_name: 'Sam', skill_level: 'I can boil an egg', dietary: 'Peanut allergy' } });
  await opening;
  assert.equal(h.els.profileName.value, 'Sam');
  assert.equal(h.els.profileSkill.value, 'I can boil an egg', 'legacy custom experience is retained');
  assert.equal(h.els.profileDietary.value, 'Peanut allergy');
  h.els.profileName.value = 'Alex';
  const saving = h.saveProfile({ preventDefault() {} });
  assert.equal(requests.at(-1).profile.cook_name, 'Alex');
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id,
    profile: { cook_name: 'Alex' }, onboarding: { complete: true } });
  await saving;
  assert.equal(h.els.profileDialog.open, false);
  assert.equal(h.state.onboarded, true);
  assert.equal(h.state.requests.size, 0);
  assert.equal(h.speech.said.length, 0, 'settings are not spoken as chat replies');
}

{
  const h = harness();
  const requests = [];
  h.state.ws = { readyState: 1, send(body) { requests.push(JSON.parse(body)); } };
  h.els.textInput.value = 'My unfinished question';
  const loading = h.loadHistory();
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, next_offset: 20,
    sessions: [{ session_id: 'first', started_at: 1000, recipe_name: '<script>pasta</script>', turn_count: 2 }] });
  await loading;
  assert.equal(h.els.historyList.children[0].children[0].textContent, '<script>pasta</script>', 'history titles use text, not HTML');
  assert.equal(h.els.moreHistory.hidden, false);
  const detail = h.loadHistoryDetail('first');
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, next_offset: 40,
    session: { recipe_name: 'Pasta' }, turns: [{ ts: 1000, user_text: 'Question', assistant_text: '<img src=x>', vision: 'a tomato' }] });
  await detail;
  assert.equal(h.els.historyDetail.hidden, false);
  assert.equal(h.els.historyTurns.children[0].children[2].children[1], '<img src=x>', 'history messages are plain text');
  assert.equal(h.els.olderHistory.hidden, false);
  const older = h.loadHistoryDetail('first', 40);
  assert.equal(requests.at(-1).offset, 40);
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, next_offset: null,
    session: { recipe_name: 'Pasta' }, turns: [{ ts: 900, user_text: 'Older question', assistant_text: 'Older reply' }] });
  await older;
  assert.equal(h.els.historyTurns.children[0].children[1].children[1], 'Older question');
  assert.equal(h.els.textInput.value, 'My unfinished question', 'history does not alter the current draft');
  assert.equal(h.els.log.children.length, 0, 'past turns do not enter the current conversation');
  assert.equal(h.speech.said.length, 0);
}

{
  const h = harness();
  await h.openProfile();
  assert.equal(h.els.profileStatus.dataset.error, 'true', 'offline profile loading has a clear error');
  assert.equal(h.els.saveProfile.disabled, true, 'offline loading cannot overwrite the saved profile with empty fields');
  await h.loadHistory();
  assert.equal(h.els.historyStatus.dataset.error, 'true');
}

{
  const h = harness();
  h.connect();
  const first = h.sockets[0];
  const firstChat = new URL(first.url).searchParams.get('chat_id');
  first.readyState = 1; first.onopen();
  h.state.onboarded = true; h.state.visionAvailable = true;
  await h.attachPhoto({ name: 'old.jpg', type: 'image/jpeg', size: 5000 });
  h.els.textInput.value = 'Old dish draft';
  h.state.timer = { id: 123, key: 'pasta#2', remaining: 60 };
  h.state.queue.push('old queued message');
  h.state.awaitingReply = true;
  h.handle({ type: 'delta', text: 'Old dish reply' });
  h.newChat();
  const next = h.sockets[1];
  assert.equal(first.closed, true);
  assert.notEqual(new URL(next.url).searchParams.get('chat_id'), firstChat);
  assert.equal(new URL(next.url).searchParams.get('client_id'), 'test', 'profile identity survives new chats');
  assert.equal(h.els.log.children.length, 0);
  assert.equal(h.els.textInput.value, '');
  assert.equal(h.state.photo, null);
  assert.equal(h.state.queue.length, 0);
  assert.equal(h.state.awaitingReply, false);
  assert.equal(h.state.timer.id, null);
  assert.ok(h.clearedIntervals.includes(123));
  assert.equal(h.timers.length, 0, 'closing the old socket does not schedule a reconnect');
  first.onmessage({ data: JSON.stringify({ type: 'response', message: 'Late old reply' }) });
  assert.equal(h.els.log.children.length, 0, 'old replies cannot enter the new dish chat');
  const anotherPage = harness(); anotherPage.connect();
  assert.notEqual(new URL(anotherPage.sockets[0].url).searchParams.get('chat_id'), firstChat, 'reopening gets a fresh chat');
}

{
  const h = harness(); h.connect();
  const socket = h.sockets[0];
  socket.onclose();
  h.timers.at(-1)();
  assert.equal(h.sockets[1].url, socket.url, 'temporary disconnect reconnects to the same chat');
  const current = h.sockets[1];
  h.windowEvents.pagehide();
  assert.equal(current.closed, true);
  assert.equal(h.state.exiting, true);
  h.windowEvents.pageshow({ persisted: true });
  assert.equal(h.state.exiting, false);
  assert.notEqual(h.sockets.at(-1).url, current.url, 'restoring a closed page from browser cache starts fresh');
}

console.log('PASS: 15 browser behavior scenarios, including ingredient previews, recipe search, fresh chats, reconnects, profiles, history, attachments, and speech.');
}

main();
