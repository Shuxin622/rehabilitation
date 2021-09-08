$(function () {
  'use strict';

  var console = window.console || { log: function () {} };
  var $image = $('#pivotImg');
  var $pivotX = $('#pivotX');
  var $pivotY = $('#pivotY');
  var $pivotHeight = $('#pivotHeight');
  var $pivotWidth = $('#pivotWidth');
  var options = {
    aspectRatio: NaN,
    viewMode: 3,
    preview: '.img-preview',
    crop: function (e) {
      $pivotX.val(Math.round(e.detail.x));
      $pivotY.val(Math.round(e.detail.y));
      $pivotHeight.val(Math.round(e.detail.height));
      $pivotWidth.val(Math.round(e.detail.width));
    }
  };

  // Cropper
  $image.on({
    ready: function (e) {
      console.log(e.type);
    },
    cropstart: function (e) {
      console.log(e.type, e.detail.action);
    },
    cropmove: function (e) {
      console.log(e.type, e.detail.action);
    },
    cropend: function (e) {
      console.log(e.type, e.detail.action);
    },
    crop: function (e) {
      console.log(e.type);
    },
    zoom: function (e) {
      console.log(e.type, e.detail.ratio);
    }
  }).cropper(options);

});
