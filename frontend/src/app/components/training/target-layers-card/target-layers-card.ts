import { Component, input, inject, OnInit, signal, computed, effect } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { ModelService } from '../../../services/model.service';
import { ModelCapabilities, BlockTopologyGroup } from '../../../services/model-capabilities.service';
import { ToastService } from '../../../services/toast';
import { FormsModule } from '@angular/forms';

export interface LayerNode {
  name: string;        // e.g. "to_q"
  pattern: string;     // e.g. ".*double_blocks\\.0\\..*to_q.*"
  selected: boolean;
}

export interface BlockInstanceNode {
  index: number;       // e.g. 0
  pattern: string;     // e.g. ".*double_blocks\\.0\\..*"
  layers: LayerNode[];
  expanded: boolean;
  selected: boolean;   // true if ALL layers are selected
  partial: boolean;    // true if SOME layers are selected
}

export interface BlockGroupNode {
  name: string;             // e.g. "double_blocks"
  pattern: string;          // e.g. ".*double_blocks.*"
  instances: BlockInstanceNode[];
  expanded: boolean;
  selected: boolean;        // true if ALL instances and layers are selected
  partial: boolean;         // true if SOME instances/layers are selected
}

@Component({
  selector: 'app-target-layers-card',
  standalone: true,
  imports: [ReactiveFormsModule, FormsModule],
  template: `
    <div class="bg-surface-mid/30 border border-surface-mid rounded-theme-xl overflow-hidden mt-4">
      <!-- Header -->
      <div class="flex items-center justify-between p-4 border-b border-surface-mid bg-surface-low/50">
        <div class="flex items-center gap-3">
          <div class="p-2 bg-brand/10 text-brand rounded-theme-md">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
              <polyline points="3.29 7 12 12 20.71 7"></polyline>
              <line x1="12" y1="22" x2="12" y2="12"></line>
            </svg>
          </div>
          <div>
            <h3 class="text-sm font-bold text-white tracking-widest uppercase">Targeted Layers</h3>
            <p class="text-xs text-text-subtle mt-0.5">Select specific blocks and modules to train. Leave all checked to train the full model.</p>
          </div>
        </div>

        <div class="flex items-center gap-4">
          <label class="flex items-center gap-2 cursor-pointer group">
             <span class="text-[10px] font-black uppercase tracking-[0.2em] transition-colors"
                   [class.text-brand]="isFilteringEnabled()"
                   [class.text-text-disabled]="!isFilteringEnabled()">
                Filter Targets
             </span>
             <div class="relative inline-flex items-center h-5 transition-colors rounded-full w-9 focus:outline-none"
                  [class.bg-brand]="isFilteringEnabled()"
                  [class.bg-surface-high]="!isFilteringEnabled()">
                <input type="checkbox"
                       [checked]="isFilteringEnabled()"
                       (change)="toggleFilteringMode()"
                       class="sr-only peer">
                <span class="inline-block w-3.5 h-3.5 transform bg-white rounded-full transition-transform duration-200"
                      [class.translate-x-5]="isFilteringEnabled()"
                      [class.translate-x-1]="!isFilteringEnabled()"></span>
             </div>
          </label>

          @if (!modules().length && isScanning()) {
            <div class="flex items-center space-x-2 text-brand">
              <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span class="text-[10px] font-black uppercase tracking-[0.2em]">Scanning...</span>
            </div>
          } @else if (!modules().length) {
            <button (click)="scanTargetLayers()"
                    [disabled]="isScanning()"
                    class="bg-surface-high hover:bg-brand text-white text-[10px] font-black uppercase tracking-[0.1em] py-1.5 px-3 rounded-theme-md transition-colors flex items-center gap-2 border border-surface-mid hover:border-brand/50 disabled:opacity-50 disabled:cursor-not-allowed">
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              Scan Model
            </button>
          } @else if (isFilteringEnabled()) {
             <div class="text-[10px] font-black text-brand bg-brand/10 border border-brand/20 px-3 py-1 rounded-full uppercase tracking-[0.1em]">
               {{ getSelectedCount() }} / {{ getTotalCount() }} Selected
             </div>
          }
        </div>
      </div>

      <!-- Content -->
      @if (modules().length && isFilteringEnabled()) {
        <div class="p-4 border-b border-surface-mid bg-surface-low flex items-center gap-2">
             <input type="text" 
                    [(ngModel)]="searchQuery" 
                    placeholder="Filter target modules... (e.g. double_blocks, to_q)"
                    class="flex-1 bg-surface-mid border border-surface-high rounded-theme-md px-4 py-2 text-sm text-white focus:outline-none focus:border-brand transition-colors">
             <button type="button" (click)="importFromClipboard()"
                     [title]="importedModules() ? 'Imported!' : 'Import module selection from clipboard (from LoRA Inspector Copy Modules)'"
                     class="shrink-0 p-2 rounded-theme-md border transition-colors"
                     [class.bg-success/20]="importedModules()"
                     [class.border-success/30]="importedModules()"
                     [class.text-success]="importedModules()"
                     [class.bg-surface-mid]="!importedModules()"
                     [class.border-surface-high]="!importedModules()"
                     [class.text-text-subtle]="!importedModules()"
                     [class.hover:text-white]="!importedModules()"
                     [class.hover:border-brand]="!importedModules()">
                 @if (importedModules()) {
                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                         <polyline points="20 6 9 17 4 12"></polyline>
                     </svg>
                 } @else {
                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                         <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                         <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                     </svg>
                 }
             </button>
        </div>
        
        <div class="p-4 max-h-[500px] overflow-y-auto custom-scrollbar flex flex-col gap-1">
           <div class="flex items-center justify-end px-2 pb-2">
             <button type="button" (click)="selectAll()" class="text-[10px] font-bold uppercase tracking-widest text-text-subtle hover:text-white transition-colors mr-3">Select All</button>
             <button type="button" (click)="deselectAll()" class="text-[10px] font-bold uppercase tracking-widest text-text-subtle hover:text-white transition-colors">Deselect All</button>
           </div>
           
           @for (group of filteredTree(); track group.name) {
               <!-- Group Level (e.g. double_blocks) -->
               <div class="flex flex-col mb-2 bg-surface-base/50 rounded-theme-md border border-surface-high overflow-hidden">
                   <div class="flex items-center gap-3 p-2 hover:bg-surface-mid/50 transition-colors">
                       <button type="button" (click)="group.expanded = !group.expanded" class="p-1 text-text-subtle hover:text-white transition-colors">
                           <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                class="transition-transform duration-200" [class.rotate-90]="group.expanded">
                             <polyline points="9 18 15 12 9 6"></polyline>
                           </svg>
                       </button>
                       <label class="flex items-center gap-3 cursor-pointer flex-1">
                         <div class="relative flex items-center">
                           <input type="checkbox"
                                  [checked]="group.selected"
                                  (change)="toggleGroup(group)"
                                  class="peer sr-only">
                           <!-- Checkbox UI -->
                           <div class="w-5 h-5 border-2 border-surface-high rounded bg-surface-mid peer-checked:bg-brand peer-checked:border-brand transition-all flex items-center justify-center"
                                [class.bg-brand]="group.partial && !group.selected"
                                [class.border-brand]="group.partial && !group.selected"
                                [class.opacity-50]="group.partial && !group.selected">
                             @if (group.selected) {
                                 <svg class="w-3 h-3 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                                   <polyline points="20 6 9 17 4 12"></polyline>
                                 </svg>
                             } @else if (group.partial) {
                                 <svg class="w-3 h-3 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                                   <line x1="5" y1="12" x2="19" y2="12"></line>
                                 </svg>
                             }
                           </div>
                         </div>
                          <div class="flex flex-col">
                              <span class="text-sm font-bold text-text-secondary">{{ group.name }}</span>
                              <div class="flex items-center gap-2">
                                <span class="text-[10px] text-text-subtle">{{ group.instances.length }} blocks</span>
                                <span class="text-[10px] font-mono font-bold" [class.text-brand]="group.selected || group.partial" [class.text-text-subtle]="!group.selected && !group.partial">{{ getGroupSelectedCount(group) }}/{{ getGroupTotalCount(group) }}</span>
                              </div>
                          </div>
                       </label>
                   </div>
                   
                   <!-- Instances (e.g. Block 0, Block 1) -->
                   @if (group.expanded) {
                       <div class="flex flex-col border-t border-surface-high bg-surface-low/30 max-h-[300px] overflow-y-auto custom-scrollbar">
                           @for (inst of group.instances; track inst.index) {
                               <div class="flex flex-col border-b border-surface-high/50 last:border-0">
                                   <div class="flex items-center gap-3 p-2 pl-8 hover:bg-surface-mid/30 transition-colors">
                                       <button type="button" (click)="inst.expanded = !inst.expanded" class="p-1 text-text-subtle hover:text-white transition-colors">
                                           <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                                class="transition-transform duration-200" [class.rotate-90]="inst.expanded">
                                             <polyline points="9 18 15 12 9 6"></polyline>
                                           </svg>
                                       </button>
                                       <label class="flex items-center gap-3 cursor-pointer flex-1">
                                           <div class="relative flex items-center">
                                             <input type="checkbox"
                                                    [checked]="inst.selected"
                                                    (change)="toggleInstance(group, inst)"
                                                    class="peer sr-only">
                                             <div class="w-4 h-4 border-2 border-surface-high rounded bg-surface-mid peer-checked:bg-brand peer-checked:border-brand transition-all flex items-center justify-center"
                                                  [class.bg-brand]="inst.partial && !inst.selected"
                                                  [class.border-brand]="inst.partial && !inst.selected"
                                                  [class.opacity-50]="inst.partial && !inst.selected">
                                                 @if (inst.selected) {
                                                     <svg class="w-2.5 h-2.5 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                                                       <polyline points="20 6 9 17 4 12"></polyline>
                                                     </svg>
                                                 } @else if (inst.partial) {
                                                     <svg class="w-2.5 h-2.5 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                                                       <line x1="5" y1="12" x2="19" y2="12"></line>
                                                     </svg>
                                                 }
                                             </div>
                                           </div>
                                            <span class="text-xs font-medium text-text-muted">Block {{ inst.index }}</span>
                                            <span class="text-[10px] font-mono ml-1" [class.text-brand]="inst.selected || inst.partial" [class.text-text-subtle]="!inst.selected && !inst.partial">{{ getInstanceSelectedCount(inst) }}/{{ inst.layers.length }}</span>
                                       </label>
                                   </div>
                                   
                                   <!-- Layers (e.g. to_q, to_k) -->
                                   @if (inst.expanded) {
                                       <div class="flex flex-col bg-surface-base/20 py-1 max-h-[200px] overflow-y-auto custom-scrollbar">
                                           @for (layer of inst.layers; track layer.name) {
                                               <label class="flex items-center gap-3 p-1.5 pl-16 hover:bg-surface-high/30 cursor-pointer group transition-colors">
                                                 <div class="relative flex items-center">
                                                   <input type="checkbox"
                                                          [checked]="layer.selected"
                                                          (change)="toggleLayer(group, inst, layer)"
                                                          class="peer sr-only">
                                                   <div class="w-3.5 h-3.5 border-2 border-surface-high rounded-sm bg-surface-mid peer-checked:bg-brand peer-checked:border-brand transition-all flex items-center justify-center">
                                                       <svg class="w-2 h-2 text-white opacity-0 peer-checked:opacity-100 transition-opacity" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                                                         <polyline points="20 6 9 17 4 12"></polyline>
                                                       </svg>
                                                   </div>
                                                 </div>
                                                 <span class="text-[11px] font-mono text-text-subtle group-hover:text-text-secondary transition-colors">{{ layer.name }}</span>
                                               </label>
                                           }
                                       </div>
                                   }
                               </div>
                           }
                       </div>
                   }
               </div>
           }
           
           @if (filteredTree().length === 0) {
              <div class="text-center py-6 text-text-subtle text-sm italic">
                  No blocks or layers match filter "{{ searchQuery() }}"
              </div>
           }
        </div>
      } @else if (modules().length && !isFilteringEnabled()) {
         <div class="p-8 text-center text-text-subtle text-sm">
             Target module filtering is disabled. The full model will be trained on all valid parameters. <br>
             <span class="text-xs italic mt-2 block opacity-70">Enable "Filter Targets" to selectively train specific blocks or layers.</span>
         </div>
      } @else if (!isScanning()) {
         <div class="p-8 text-center text-text-subtle text-sm">
             Targetable layers have not been scanned for this model definition yet. <br>
             Click "Scan Model" above to introspect the network graph and discover targetable blocks. <br>
             <span class="text-xs italic mt-2 block opacity-70">(This will be permanently persisted to the definition YAML)</span>
         </div>
      }
    </div>
  `
})
export class TargetLayersCardComponent implements OnInit {
  definitionId = input.required<string>();
  control = input.required<FormControl<string[]>>();

  private modelService = inject(ModelService);
  private toast = inject(ToastService);

  modules = signal<string[]>([]);
  topology = signal<BlockTopologyGroup[]>([]);
  isScanning = signal(false);
  isFilteringEnabled = signal(false);
  searchQuery = signal('');
  importedModules = signal(false);

  // Internal state representation of the hierarchy
  tree = signal<BlockGroupNode[]>([]);

  filteredTree = computed(() => {
    const q = this.searchQuery().toLowerCase().trim();
    const fullTree = this.tree();
    if (!q) return fullTree;

    // Deep filter
    return fullTree.map(group => {
      const matchGroup = group.name.toLowerCase().includes(q);
      const filteredInstances = group.instances.map(inst => {
        const matchInst = matchGroup || inst.index.toString() === q;
        const filteredLayers = inst.layers.filter(l => matchInst || l.name.toLowerCase().includes(q));
        return { ...inst, layers: filteredLayers };
      }).filter(inst => matchGroup || inst.layers.length > 0);

      return { ...group, instances: filteredInstances, expanded: true };
    }).filter(group => group.instances.length > 0);
  });

  constructor() {
    effect(() => {
      const defId = this.definitionId();
      if (defId) {
        this.loadCapabilities();
      }
    });
  }

  ngOnInit() {
    // initial load is handled by the effect, but kept for interface compliance
  }

  private loadCapabilities() {
    if (!this.definitionId()) return;
    this.modelService.getCapabilities(this.definitionId()).subscribe({
      next: (caps: ModelCapabilities) => {
        this.modules.set(caps.lora_targetable_modules || []);
        this.topology.set(caps.block_topology || []);
        this.buildTree();
      },
      error: (err: any) => {
        console.error('Failed to load capabilities', err);
      }
    });
  }

  scanTargetLayers() {
    if (!this.definitionId() || this.isScanning()) return;
    this.isScanning.set(true);

    this.modelService.enrichDefinition(this.definitionId()).subscribe({
      next: (res) => {
        this.modules.set(res.lora_targetable_modules || []);
        this.topology.set(res.block_topology || []);
        this.isScanning.set(false);
        this.toast.success('Successfully scanned target layers.');
        this.buildTree();
      },
      error: (err: any) => {
        this.isScanning.set(false);
        this.toast.error('Failed to scan model modules.');
      }
    });
  }

  // --- Tree Construction & Syncing ---

  /**
   * Public: rebuild the tree from the current FormControl value.
   * Called by parent after externally patching targeted_layers (job reload, template apply).
   */
  refreshFromControl() {
    this.buildTree();
  }

  /**
   * Builds the hierarchical truth state.
   * Combines block_topology (groups & counts) with lora_targetable_modules (layer templates).
   */
  private buildTree() {
    const mods = this.modules();
    const tops = this.topology();

    if (!mods.length || !tops.length) {
      this.tree.set([]);
      return;
    }

    // Parse the initial forms value to see what's checked
    const formPatterns = this.control().value || [];

    // Auto-enable filtering if formPatterns has explicitly filtered items
    if (formPatterns.length > 0) {
      this.isFilteringEnabled.set(true);
    }

    // Treat fully empty or literal '.*' as "all selected"
    const isAllSelected = formPatterns.length === 0 || formPatterns.includes('.*');
    const isNoneSelected = formPatterns.length > 0 && formPatterns[0] === '__none__';

    const newTree: BlockGroupNode[] = [];

    for (const top of tops) {
      const instances: BlockInstanceNode[] = [];
      let groupSelected = true;
      let groupPartial = false;

      for (let i = 0; i < top.count; i++) {
        const instPattern = `.*${top.name}\\\\.${i}\\\\..*`;
        const layers: LayerNode[] = [];
        let instSelected = true;
        let instPartial = false;

        for (const mod of mods) {
          const layerPattern = `.*${top.name}\\\\.${i}\\\\..*${mod}.*`;

          // Determine selection state based on existing regexes
          let lSelected = false;
          if (isAllSelected) {
            lSelected = true;
          } else if (isNoneSelected) {
            lSelected = false;
          } else {
            // Check if ANY form pattern covers this layer
            lSelected = formPatterns.some(fp => {
              // If they have the exact exact pattern
              if (fp === layerPattern) return true;
              // If they selected the whole instance (.*double_blocks\.0\..*)
              if (fp === instPattern) return true;
              // If they selected the whole group (.*double_blocks.*)
              if (fp === `.*${top.name}.*`) return true;
              // If they selected this module globally across all blocks (.*to_q.*)
              if (fp === `.*${mod}.*`) return true;
              return false;
            });
          }

          if (!lSelected) instSelected = false;
          if (lSelected) instPartial = true;

          layers.push({
            name: mod,
            pattern: layerPattern,
            selected: lSelected
          });
        }

        if (!instSelected) groupSelected = false;
        if (instPartial) groupPartial = true;

        instances.push({
          index: i,
          pattern: instPattern,
          layers: layers,
          expanded: false,
          selected: instSelected,
          partial: instSelected ? false : instPartial
        });
      }

      newTree.push({
        name: top.name,
        pattern: `.*${top.name}.*`,
        instances: instances,
        expanded: false,
        selected: groupSelected,
        partial: groupSelected ? false : groupPartial
      });
    }

    this.tree.set(newTree);

    if (formPatterns.length === 0 && tops.length > 0 && mods.length > 0) {
      this.syncStateToControl();
    }
  }

  /**
   * Flattens the tree state into the optimal set of regexes.
   * We try to compress (e.g. if all blocks are selected, output .*group.*).
   */
  private syncStateToControl() {
    const patterns: string[] = [];
    let allGloballySelected = true;

    for (const group of this.tree()) {
      if (group.selected) {
        patterns.push(group.pattern);
      } else {
        allGloballySelected = false;
        for (const inst of group.instances) {
          if (inst.selected) {
            patterns.push(inst.pattern);
          } else if (inst.partial) {
            for (const layer of inst.layers) {
              if (layer.selected) {
                patterns.push(layer.pattern);
              }
            }
          }
        }
      }
    }

    if (allGloballySelected) {
      this.control().setValue([]); // Empty implies all
    } else if (patterns.length === 0) {
      this.control().setValue(['__none__']);
    } else {
      this.control().setValue(patterns);
    }

    this.control().markAsDirty();
  }

  // --- Toggles ---

  private updateCascadingState(tree: BlockGroupNode[]) {
    for (const group of tree) {
      let groupAll = true;
      let groupAny = false;

      for (const inst of group.instances) {
        let instAll = true;
        let instAny = false;

        for (const layer of inst.layers) {
          if (!layer.selected) instAll = false;
          if (layer.selected) instAny = true;
        }

        inst.selected = instAll;
        inst.partial = !instAll && instAny;

        if (!instAll) groupAll = false;
        if (instAny) groupAny = true;
      }

      group.selected = groupAll;
      group.partial = !groupAll && groupAny;
    }
  }

  toggleGroup(group: BlockGroupNode) {
    const newState = !group.selected;
    group.selected = newState;
    group.partial = false;
    for (const inst of group.instances) {
      inst.selected = newState;
      inst.partial = false;
      for (const layer of inst.layers) {
        layer.selected = newState;
      }
    }
    this.updateCascadingState(this.tree());
    this.syncStateToControl();
  }

  toggleInstance(group: BlockGroupNode, inst: BlockInstanceNode) {
    const newState = !inst.selected;
    inst.selected = newState;
    inst.partial = false;
    for (const layer of inst.layers) {
      layer.selected = newState;
    }
    this.updateCascadingState(this.tree());
    this.syncStateToControl();
  }

  toggleLayer(group: BlockGroupNode, inst: BlockInstanceNode, layer: LayerNode) {
    layer.selected = !layer.selected;
    this.updateCascadingState(this.tree());
    this.syncStateToControl();
  }

  toggleFilteringMode() {
    const current = this.isFilteringEnabled();
    this.isFilteringEnabled.set(!current);

    if (!this.isFilteringEnabled()) {
      // If they disabled it, set the control to [] so everything trains
      this.control().setValue([]);
      this.control().markAsDirty();
      // We can also select all in the UI tree so it looks checked if they turn it back on
      for (const g of this.tree()) {
        let gSelected = true;
        for (const inst of g.instances) {
          let instSelected = true;
          for (const layer of inst.layers) {
            layer.selected = true;
          }
          inst.selected = instSelected;
          inst.partial = false;
        }
        g.selected = gSelected;
        g.partial = false;
      }
    } else {
      // When turning ON, sync current tree state to control.
      // (If they hadn't touched it, it will sync [] meaning everything is checked).
      this.syncStateToControl();
    }
  }

  selectAll() {
    for (const g of this.tree()) {
      this.toggleGroup({ ...g, selected: false });
    }
  }

  deselectAll() {
    for (const g of this.tree()) {
      this.toggleGroup({ ...g, selected: true });
    }
  }

  getSelectedCount(): number {
    let count = 0;
    for (const g of this.tree()) {
      for (const i of g.instances) {
        for (const l of i.layers) {
          if (l.selected) count++;
        }
      }
    }
    return count;
  }

  getTotalCount(): number {
    let count = 0;
    for (const g of this.tree()) {
      for (const i of g.instances) {
        count += i.layers.length;
      }
    }
    return count;
  }

  getGroupSelectedCount(group: BlockGroupNode): number {
    let count = 0;
    for (const i of group.instances) {
      for (const l of i.layers) {
        if (l.selected) count++;
      }
    }
    return count;
  }

  getGroupTotalCount(group: BlockGroupNode): number {
    let count = 0;
    for (const i of group.instances) {
      count += i.layers.length;
    }
    return count;
  }

  getInstanceSelectedCount(inst: BlockInstanceNode): number {
    return inst.layers.filter(l => l.selected).length;
  }

  /** Import module selection from clipboard (from LoRA Inspector "Copy Modules"). */
  async importFromClipboard() {
    try {
      const text = await navigator.clipboard.readText();
      if (!text?.trim()) {
        this.toast.error('Clipboard is empty');
        return;
      }

      // Parse module names â€” support JSON array or comma-separated
      let moduleNames: string[] = [];
      const trimmed = text.trim();
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) {
          moduleNames = parsed.map((m: any) => String(m).trim()).filter(Boolean);
        }
      } catch {
        // Fall back to comma-separated
        moduleNames = trimmed.split(/[,\n]+/).map(s => s.trim().replace(/["'\[\]]/g, '')).filter(Boolean);
      }

      if (!moduleNames.length) {
        this.toast.error('No module names found in clipboard');
        return;
      }

      // Expand BFL/ComfyUI merged modules to diffusers equivalents
      // so e.g. "qkv" matches "to_q", "to_k", "to_v" in model scan
      const BFL_EXPAND: Record<string, string[]> = {
        // Flux2/Klein BFL â†’ diffusers (both fused and unfused variants)
        'qkv': ['to_qkv', 'to_q', 'to_k', 'to_v'],
        'proj': ['to_out.0', 'to_out'],
        'img_attn.qkv': ['to_qkv', 'to_q', 'to_k', 'to_v'],
        'img_attn.proj': ['to_out.0', 'to_out'],
        'txt_attn.qkv': ['to_added_qkv', 'add_q_proj', 'add_k_proj', 'add_v_proj'],
        'txt_attn.proj': ['to_add_out'],
        'img_mlp.0': ['ff.net.0.proj', 'ff.linear_in'],
        'img_mlp.2': ['ff.net.2', 'ff.linear_out'],
        'txt_mlp.0': ['ff_context.net.0.proj', 'ff_context.linear_in'],
        'txt_mlp.2': ['ff_context.net.2', 'ff_context.linear_out'],
        'linear1': ['linear1', 'to_qkv_mlp_proj'],
        'linear2': ['linear2', 'to_out'],
        // Flux2 dev single blocks (diffusers naming)
        'to_qkv_mlp_proj': ['to_qkv_mlp_proj', 'linear1'],
        // Diffusers fused â†’ pre-fusion PEFT targets
        'to_qkv': ['to_q', 'to_k', 'to_v'],
        'to_added_qkv': ['add_q_proj', 'add_k_proj', 'add_v_proj'],
        'odd_q_proj': ['to_q'],
        'odd_k_proj': ['to_k'],
        'odd_v_proj': ['to_v'],
      };

      const tree = this.tree();
      if (!tree.length) {
        this.toast.error('Scan model target layers first');
        return;
      }

      // Enable filtering mode
      if (!this.isFilteringEnabled()) {
        this.isFilteringEnabled.set(true);
      }

      // Detect if input is full module paths (per-instance, from inspector)
      // vs short module type patterns (suffix-only, legacy).
      // Full paths contain dot-separated segments with numeric block indices.
      const isFullPaths = moduleNames.some(m => /\.\d+\./.test(m) && m.split('.').length > 3);

      if (isFullPaths) {
        // â”€â”€ Per-instance matching â”€â”€
        // LoRA paths like "diffusion_model.double_blocks.0.img_attn.qkv"
        // must be matched to tree nodes using topology group names.

        // Map LoRA attr path segments â†’ topology group names
        const BLOCK_ALIASES: Record<string, string> = {
          'double_blocks': 'double_blocks',
          'single_blocks': 'single_blocks',
          'transformer_blocks': 'double_blocks',
          'single_transformer_blocks': 'single_blocks',
          'down_blocks': 'down_blocks',
          'mid_block': 'mid_block',
          'up_blocks': 'up_blocks',
        };

        // Parse each module path by splitting on dots
        const parsedMods: { group: string; idx: number; suffix: string }[] = [];
        for (const m of moduleNames) {
          const parts = m.split('.');
          for (let si = 0; si < parts.length - 1; si++) {
            const alias = BLOCK_ALIASES[parts[si]];
            if (!alias) continue;
            const idx = parseInt(parts[si + 1], 10);
            if (isNaN(idx)) continue;
            const suffix = parts.slice(si + 2).join('.');
            if (suffix) parsedMods.push({ group: alias, idx, suffix });
            break;
          }
        }

        // Expand fused/BFL suffixes â†’ pre-fusion layer names
        const FUSED: Record<string, string[]> = {
          'img_attn.qkv': ['to_q', 'to_k', 'to_v'],
          'img_attn.proj': ['to_out', 'to_out.0'],
          'txt_attn.qkv': ['add_q_proj', 'add_k_proj', 'add_v_proj'],
          'txt_attn.proj': ['to_add_out'],
          'img_mlp.0': ['ff.net.0.proj', 'ff.linear_in'],
          'img_mlp.2': ['ff.net.2', 'ff.linear_out'],
          'txt_mlp.0': ['ff_context.net.0.proj', 'ff_context.linear_in'],
          'txt_mlp.2': ['ff_context.net.2', 'ff_context.linear_out'],
          'linear1': ['linear1', 'to_qkv_mlp_proj', 'attn.to_qkv_mlp_proj'],
          'linear2': ['linear2', 'to_out', 'attn.to_out'],
          'to_qkv': ['to_q', 'to_k', 'to_v'],
          'to_added_qkv': ['add_q_proj', 'add_k_proj', 'add_v_proj'],
          'to_qkv_mlp_proj': ['to_q', 'to_k', 'to_v', 'ff.net.0.proj'],
        };

        // Build lookup: groupName â†’ blockIdx â†’ Set<layerName>
        const lookup = new Map<string, Map<number, Set<string>>>();
        for (const p of parsedMods) {
          const expanded = FUSED[p.suffix] || [p.suffix];
          if (!lookup.has(p.group)) lookup.set(p.group, new Map());
          const idxMap = lookup.get(p.group)!;
          if (!idxMap.has(p.idx)) idxMap.set(p.idx, new Set());
          const names = idxMap.get(p.idx)!;
          for (const s of expanded) names.add(s);
        }

        let matchCount = 0;
        for (const group of tree) {
          const idxMap = lookup.get(group.name);
          if (!idxMap) {
            for (const inst of group.instances) {
              for (const layer of inst.layers) layer.selected = false;
            }
            continue;
          }
          for (const inst of group.instances) {
            const names = idxMap.get(inst.index);
            for (const layer of inst.layers) {
              const hit = names ? names.has(layer.name) : false;
              layer.selected = hit;
              if (hit) matchCount++;
            }
          }
        }

        this.updateCascadingState(tree);
        this.syncStateToControl();

        if (matchCount > 0) {
          this.toast.success(`Imported ${moduleNames.length} modules â€” ${matchCount} layers selected`);
          this.importedModules.set(true);
          setTimeout(() => this.importedModules.set(false), 2000);
        } else {
          this.toast.error(`No matching layers found for ${moduleNames.length} module paths`);
        }
      } else {
        // â”€â”€ Legacy: expand BFL/fused names and match by layer.name suffix â”€â”€
        const expandedNames: Set<string> = new Set(moduleNames);
        for (const m of moduleNames) {
          const expanded = BFL_EXPAND[m];
          if (expanded) expanded.forEach((e: string) => expandedNames.add(e));
        }

        let matchCount = 0;
        for (const group of tree) {
          for (const inst of group.instances) {
            for (const layer of inst.layers) {
              const matches = Array.from(expandedNames).some(m =>
                layer.name === m || layer.name.includes(m) || m.includes(layer.name)
              );
              layer.selected = matches;
              if (matches) matchCount++;
            }
          }
        }

        this.updateCascadingState(tree);
        this.syncStateToControl();

        if (matchCount > 0) {
          this.toast.success(`Imported ${moduleNames.length} modules â€” ${matchCount} layers selected`);
          this.importedModules.set(true);
          setTimeout(() => this.importedModules.set(false), 2000);
        } else {
          this.toast.error(`No matching layers found for: ${moduleNames.join(', ')}`);
        }
      }
    } catch (err) {
      this.toast.error('Failed to read clipboard');
    }
  }
}

