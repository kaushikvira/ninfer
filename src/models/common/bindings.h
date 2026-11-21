#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "models/common/weights.h"

#include <map>
#include <span>

namespace ninfer::models {

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

void require_shared_binding(const artifact::ParameterReference&,
                            const artifact::ParameterReference&);

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {});
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  std::optional<QType> format = QType::BF16);
    // Binds a direct tensor in the representation the artifact declares, limited to the formats
    // the consuming operations can execute. The declaration is never overridden and never
    // reinterpreted: a declared representation outside `accepted` fails the binding.
    [[nodiscard]] WeightId direct_as_declared(std::string name, artifact::Shape shape,
                                              std::span<const QType> accepted);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    std::map<std::string, WeightId, std::less<>> parameters_;
};

[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized,
                std::span<const Device> devices);

} // namespace ninfer::models
