<script>
export default {
  name: "InformationSourceLocalization",
  data() {
    return {};
  },
  mounted() {
    setTimeout(() => {
      this.$refs.text.innerHTML = this.text;
    });
  },
  methods: {
    setHighlighting(start, end) {
      let lines = [start, end];
      if (!end) lines = [start];
      if (lines.length > 1) {
        lines = [...Array(lines[1] - lines[0] + 1).keys()].map(
          (i) => i + lines[0]
        );
      }

      let highlightedText = this.text.split("\n");
      lines.forEach((line, index) => {
        highlightedText[line] = `<span ${
          index === 0 ? 'id="firstLine"' : ""
        } highlight="true">${highlightedText[line]}</span>`;
      });
      this.$refs.text.innerHTML = highlightedText.join("\n");
      this.$nextTick(() => {
        let firstLine = document.getElementById("firstLine");
        if (firstLine) {
          firstLine.scrollIntoView({
            behavior: 'auto',
            block: 'center',
            inline: 'center'
          });
        }
      });
    },
    clearHighlighting() {
      this.setHighlighting(-1, -1);
    },
    highlightLine(props) {
      let rowIndex = props.rowIndex;
      let lines = this.rows[rowIndex].lines;
      console.log(lines);

      this.clearHighlighting();
      this.setHighlighting(lines[0], lines[1]);
      setTimeout(() => {
        this.clearHighlighting();
      }, 10000);
    },
    // following method is REQUIRED
    // (don't change its name --> "show")
    show() {
      this.$refs.dialog.show();
    },

    // following method is REQUIRED
    // (don't change its name --> "hide")
    hide() {
      this.$refs.dialog.hide();
    },

    onDialogHide() {
      // required to be emitted
      // when QDialog emits "hide" event
      this.$emit("hide");
    },

    onOKClick() {
      // on OK, it is REQUIRED to
      // emit "ok" event (with optional payload)
      // before hiding the QDialog
      this.$emit("ok");
      // or with payload: this.$emit('ok', { ... })

      // then hiding dialog
      this.hide();
    },

    onCancelClick() {
      // we just need to hide the dialog
      this.hide();
    },
  },

  props: {
    rows: [],
    columns: [],
    text: String,
  },

  emits: [
    // REQUIRED
    "ok",
    "hide",
  ],
};
</script>

<template>
  <q-dialog class="full-width full-height" ref="dialog" @hide="onDialogHide">
    <q-card
      class="q-dialog-plugin full-width full-height q-pa-md column"
      style="max-width: unset; gap: 10px"
    >
      <div class="flex col" style="gap: 10px">
        <q-card bordered class="relative-position full-height bg-grey-2 col rounded-borders">
          <div
            style="white-space: pre-line; overflow: scroll; height: 100%"
            ref="text"
          ></div>
<!--          <div class="blur-top">-->

<!--          </div>-->
<!--          <div class="blur-bottom">-->

<!--          </div>-->
        </q-card>
        <q-card bordered class="bg-grey-2 col">
          <h6 class="text-h6 q-my-sm q-pa-md">Click on row to see source</h6>

          <q-table
            class="col col-grow"
            title="Medications"
            :rows="rows"
            :columns="columns"
            row-key="name"
            :rows-per-page-options="[0, 10, 20, 30]"
          >
            <template v-slot:body="props">
              <q-tr :props="props" @click="highlightLine(props)">
                <q-td key="name" :props="props">
                  {{ props.row.name }}
                </q-td>
                <q-td key="dose" :props="props">
                  {{ props.row.dose }}
                </q-td>
                <q-td key="frequency" :props="props">
                  {{ props.row.frequency }}
                </q-td>
                <q-td key="route" :props="props">
                  {{ props.row.route }}
                </q-td>
                <q-td key="lines" :props="props">
                  {{ props.row.lines }}
                </q-td>
              </q-tr>
            </template>
          </q-table>
        </q-card>
      </div>
      <q-card-actions align="right">
        <q-btn color="primary" label="OK" @click="onOKClick" />
      </q-card-actions>
    </q-card>
  </q-dialog>
</template>

<style lang="scss">
span[highlight="true"] {
  background-color: rgba(116, 198, 232, 0.4);
  padding: 1px;
  border-radius: 1em;
  font-size: 1.1em;
  font-weight: bold;
}

.blur-top{
  position: absolute;
  top: 0;
  width: 100%;
  height: 300px;
  backdrop-filter: blur(22px);
}
.blur-bottom{
  position: absolute;
  bottom: 0;
  width: 100%;

  height: 300px;
  backdrop-filter: blur(22px);
}
</style>
