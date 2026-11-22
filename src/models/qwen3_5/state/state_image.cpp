#include "models/qwen3_5/state/state_image.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5 {

StateImageDeviceLayout plan_state_image_layout(const StateImageSpec& spec) {
    StateImageBuilder builder;
    StateImageDeviceLayout layout;
    layout.linear              = plan_linear_attention_state(builder, spec.linear);
    layout.continuation_hidden = builder.add(DType::BF16, {spec.hidden}, 1, "Continuation hidden");
    // A masked-draft profile that declares NO local (windowed) draft layers carries its drafter
    // state entirely in the paged-context half (`DFlashPersistentSpec::full_layers`), so it names
    // no cyclic local cache. `plan_cyclic_kv_cache` refuses `layers == 0` ("Cyclic KV geometry is
    // invalid", core/cyclic_kv_cache.cpp:25), which is the right answer for a cache that does not
    // exist - it is not the right answer for a profile that has a drafter at all, and that is what
    // the DSpine arm of program/planning/startup.cpp:96-115 describes (its comment: "it takes the
    // paged-context half of the masked-draft state and no cyclic local state at all", with
    // `.layers = 0U`). Gated on the layers the spec names, so every profile that can reach this
    // line today (DFlash/DFlash2 name `draft.local_layer_count()`) is unaffected byte for byte.
    if (spec.dflash_local && spec.dflash_local->layers > 0) {
        const auto& local   = *spec.dflash_local;
        layout.dflash_local = plan_cyclic_kv_cache(builder, local.layers, local.capacity,
                                                   local.kv_heads, local.head_dim);
    }
    layout.images = builder.finish();
    return layout;
}

StateImageViews::StateImageViews(MemorySpan backing, const StateImageDeviceLayout& layout)
    : storage_(backing, layout.images), linear_(storage_, layout.linear),
      hidden_(layout.continuation_hidden) {
    if (layout.dflash_local) {
        dflash_local_.emplace(storage_, *layout.dflash_local);
        dflash_group_ = layout.dflash_local->group;
    }
}

Tensor StateImageViews::continuation_hidden(std::int64_t offset) const {
    return storage_.tensor(hidden_, offset);
}

StateFieldView StateImageViews::continuation_hidden(const Tensor& offsets) const {
    return storage_.field(hidden_, offsets);
}

CyclicKVCacheView* StateImageViews::dflash_local() noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

const CyclicKVCacheView* StateImageViews::dflash_local() const noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

TransferWork StateImageViews::dflash_local_transfer_work() const {
    if (!dflash_group_) { throw std::logic_error("StateImage has no DFlash local state"); }
    return {.payload_bytes = dflash_group_->bytes, .copy_operations = 1};
}

} // namespace ninfer::models::qwen3_5
