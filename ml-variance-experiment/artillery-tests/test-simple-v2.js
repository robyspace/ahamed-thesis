module.exports = {
  setTestPayload: function(userContext, events, done) {
    userContext.vars.testData = {
      id: Date.now(),
      message: "test"
    };
    return done();
  }
};
